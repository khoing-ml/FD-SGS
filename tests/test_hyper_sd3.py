"""CPU checks for SD3 flow-scheduler sampling without downloading model weights."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from fd_sgs.hyper_sd3 import HyperSD3SamplingConfig, sample


class ToyTransformer:
    dtype = torch.float32
    config = SimpleNamespace(in_channels=4, patch_size=1)

    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, timestep, encoder_hidden_states, pooled_projections,
                 return_dict=False):
        assert len(hidden_states) == len(timestep) == len(encoder_hidden_states) == len(pooled_projections)
        self.calls.append(len(hidden_states))
        condition = encoder_hidden_states.mean((1, 2)).view(-1, 1, 1, 1)
        return (.1 * hidden_states + .02 * condition,)


class ToyPipe:
    _execution_device = torch.device("cpu")

    def __init__(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
        self.transformer = ToyTransformer()
        self.vae = SimpleNamespace(dtype=torch.float32,
                                   config=SimpleNamespace(scaling_factor=.5, shift_factor=.1),
                                   decode=lambda z, return_dict=False: (z,))
        self.image_processor = SimpleNamespace(postprocess=lambda x, output_type: list(x.unbind()))
        self.freed = False

    def encode_prompt(self, **kwargs):
        pos = torch.ones(1, 2, 3)
        neg = torch.zeros_like(pos)
        return pos, neg, torch.ones(1, 2), torch.zeros(1, 2)

    def prepare_latents(self, n, channels, h, w, dtype, device, generator, latents):
        return torch.randn((n, channels, h // 8, w // 8), generator=generator, device=device, dtype=dtype)

    def maybe_free_model_hooks(self):
        self.freed = True


def reward(images, prompt):
    return [float(x.mean()) for x in images]


def cfg(steps=16, **kwargs):
    guidance_steps = (1, 2, 3) if steps == 4 else (1, 2, 3, 4, 5, 6)
    return HyperSD3SamplingConfig(steps=steps, height=16, width=16, particles=2,
                                  guidance_steps=guidance_steps,
                                  cfg_scale={4: 3.0, 8: 5.0, 16: 7.0}[steps], **kwargs)


@pytest.mark.parametrize("steps", [4, 8, 16])
def test_unguided_matches_native_flow_euler(steps):
    pipe = ToyPipe()
    actual, stats = sample(pipe, "test", reward, cfg(steps, method="unguided"))
    reference = ToyPipe()
    reference.scheduler.set_timesteps(steps, device="cpu")
    x = reference.prepare_latents(2, 4, 16, 16, torch.float32, "cpu", torch.Generator().manual_seed(42), None)
    for t in reference.scheduler.timesteps:
        scale = {4: 3.0, 8: 5.0, 16: 7.0}[steps]
        pred = .1 * x + .02 * scale
        x = reference.scheduler.step(pred, t, x).prev_sample
    torch.testing.assert_close(torch.stack(actual), x / .5 + .1)
    assert stats["sequential_nfe"] == steps
    assert stats["rollout_denoiser_forward_calls"] == 0
    assert pipe.freed


@pytest.mark.parametrize("twin_sampler", ["flow-grpo", "fdfo"])
def test_stochastic_probes_score_completed_sd3_states(twin_sampler):
    baseline, _ = sample(ToyPipe(), "test", reward, cfg(method="unguided"))
    previews = []
    guided, stats = sample(ToyPipe(), "test", reward, cfg(method="independent-fd", twin_sampler=twin_sampler),
                           on_step=lambda trace, images: previews.append(images) if images is not None else None)
    assert len(previews) == 6
    torch.testing.assert_close(torch.stack(previews[0][:2]), torch.stack(baseline))
    assert stats["steps"][0]["guided"]
    assert stats["reward_calls"] == 7
    assert not torch.allclose(torch.stack(guided), torch.stack(baseline))
    zero_cfg = (dict(noise_level=0) if twin_sampler == "flow-grpo" else dict(exploration=0))
    zero, stats = sample(ToyPipe(), "test", reward,
                         cfg(method="independent-fd", twin_sampler=twin_sampler, **zero_cfg))
    torch.testing.assert_close(torch.stack(zero), torch.stack(baseline))
    assert all(not trace["guided"] for trace in stats["steps"])


def test_cli_sd3_defaults_and_rejects_ddim(monkeypatch, capsys):
    import json
    import sys
    from fd_sgs.cli import main

    argv = ["fd-sgs", "--prompt", "test", "--backbone", "hyper-sd3", "--reward", "brightness", "--dry-run"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    config = json.loads(capsys.readouterr().out)["configs"][0]
    assert config["steps"] == 16
    assert config["cfg_scale"] == 7.0
    assert config["twin_sampler"] == "flow-grpo"
    monkeypatch.setattr(sys, "argv", argv + ["--twin-sampler", "ddim"])
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("steps", [4, 8, 16])
def test_sd3_loader_selects_matching_lora_and_small_fusion_scale(monkeypatch, steps):
    import argparse
    import importlib.util
    from diffusers import StableDiffusion3Pipeline
    from fd_sgs.cli import _load_pipeline
    from fd_sgs.hyper_sd import HYPER_LORAS, HYPER_MODELS

    calls = []
    pipe = ToyPipe()
    pipe.vae.enable_slicing = lambda: None
    pipe.vae.enable_tiling = lambda: None
    pipe.load_lora_weights = lambda repo, weight_name: calls.append((repo, weight_name))
    pipe.fuse_lora = lambda lora_scale: calls.append(("fused", lora_scale))
    pipe.to = lambda device: pipe
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(StableDiffusion3Pipeline, "from_pretrained", lambda model, **kw: (
        calls.append((model, kw)) or pipe))
    args = argparse.Namespace(backbone="hyper-sd3", steps=steps, model=HYPER_MODELS["hyper-sd3"],
                              dtype="float16", hyper_lora_repo="ByteDance/Hyper-SD", lora_scale=.125,
                              offload=False, device="cpu")
    assert _load_pipeline(args) is pipe
    assert calls[1] == ("ByteDance/Hyper-SD", HYPER_LORAS[("hyper-sd3", steps)])
    assert calls[2] == ("fused", .125)


@pytest.mark.parametrize("bad", [dict(steps=12), dict(twin_sampler="ddim"), dict(guidance_steps=(17,))])
def test_config_rejects_unsupported_sd3_settings(bad):
    with pytest.raises(ValueError):
        HyperSD3SamplingConfig(**bad)


def test_four_step_config_chooses_valid_guidance_steps():
    assert HyperSD3SamplingConfig(steps=4).guidance_steps == (1, 2, 3, 4)
