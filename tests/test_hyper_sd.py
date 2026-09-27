"""CPU checks of the discrete DDIM path; no model download required."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import DDIMScheduler

from fd_sgs.hyper_sd import HyperSamplingConfig, sample


class ToyUNet:
    dtype = torch.float32
    config = SimpleNamespace(in_channels=4)

    def __init__(self):
        self.calls = []

    def __call__(self, x, t, encoder_hidden_states, added_cond_kwargs=None, return_dict=False):
        assert encoder_hidden_states.shape[0] == len(x)
        if added_cond_kwargs is not None:
            assert added_cond_kwargs["text_embeds"].shape[0] == len(x)
            assert added_cond_kwargs["time_ids"].shape[0] == len(x)
        self.calls.append(len(x))
        condition = encoder_hidden_states.mean((1, 2)).view(-1, 1, 1, 1)
        return (x * .1 + condition * .02,)


class ToyVAE:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=.5, force_upcast=False)

    def decode(self, z, return_dict=False):
        return (z,)


class ToyPipe:
    _execution_device = torch.device("cpu")

    def __init__(self, sdxl=False):
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, timestep_spacing="trailing")
        self.unet = ToyUNet()
        self.vae = ToyVAE()
        self.image_processor = SimpleNamespace(postprocess=lambda x, output_type: list(x.unbind()))
        self.text_encoder_2 = SimpleNamespace(config=SimpleNamespace(projection_dim=2)) if sdxl else None
        self.freed = False

    def encode_prompt(self, **kwargs):
        pos = torch.ones(1, 2, 3)
        neg = torch.zeros_like(pos)
        if self.text_encoder_2 is not None:
            return pos, neg, torch.ones(1, 2), torch.zeros(1, 2)
        return pos, neg

    def _get_add_time_ids(self, *args, **kwargs):
        return torch.ones(1, 6)

    def prepare_latents(self, n, channels, h, w, dtype, device, generator, latents):
        return torch.randn((n, channels, h // 8, w // 8), generator=generator, device=device, dtype=dtype)

    def maybe_free_model_hooks(self):
        self.freed = True


def reward(images, prompt):
    return [float(x.mean()) for x in images]


def cfg(**kwargs):
    return HyperSamplingConfig(backbone="hyper-sd15", height=16, width=16, particles=2, **kwargs)


def test_unguided_matches_explicit_ddim_cfg_loop():
    pipe = ToyPipe()
    actual, stats = sample(pipe, "test", reward, cfg(method="unguided"))
    reference = ToyPipe()
    reference.scheduler.set_timesteps(8, device="cpu")
    x = reference.prepare_latents(2, 4, 16, 16, torch.float32, "cpu", torch.Generator().manual_seed(42), None)
    for t in reference.scheduler.timesteps:
        uncond = .1 * x
        cond = .1 * x + .02
        predicted = uncond + 5 * (cond - uncond)
        x = reference.scheduler.step(predicted, t, x, eta=0).prev_sample
    torch.testing.assert_close(torch.stack(actual), x / .5)
    assert stats["sequential_nfe"] == 8
    assert stats["rollout_denoiser_forward_calls"] == 0
    assert pipe.freed


@pytest.mark.parametrize("backbone,steps", [("hyper-sd15", 8), ("hyper-sd15", 12),
                                            ("hyper-sdxl", 8), ("hyper-sdxl", 12)])
def test_probe_rollout_and_zero_eta(backbone, steps):
    base = dict(backbone=backbone, height=16, width=16, particles=2)
    baseline, _ = sample(ToyPipe(backbone == "hyper-sdxl"), "test", reward,
                         HyperSamplingConfig(method="unguided", steps=steps, **base))
    previews = []
    guided, stats = sample(ToyPipe(backbone == "hyper-sdxl"), "test", reward,
                           HyperSamplingConfig(method="independent-fd", steps=steps, **base),
                           on_step=lambda trace, images: previews.append(images) if images is not None else None)
    assert len(previews) == 6
    # Before the first correction, the anchor's deterministic rollout is the
    # same complete image as an unguided eight-step run.
    torch.testing.assert_close(torch.stack(previews[0][:2]), torch.stack(baseline))
    assert stats["steps"][0]["guided"]
    rollout_levels = sum(steps - guided_step for guided_step in range(1, 7))
    assert stats["rollout_denoiser_forward_calls"] == rollout_levels * 4  # four states, one per UNet call
    assert stats["rollout_denoiser_batch_elements"] == rollout_levels * 4
    assert stats["denoiser_forward_calls"] == steps * 2
    assert stats["denoiser_batch_elements"] == steps * 2
    assert stats["reward_calls"] == 7
    assert not torch.allclose(torch.stack(guided), torch.stack(baseline))
    zero, zero_stats = sample(ToyPipe(backbone == "hyper-sdxl"), "test", reward,
                              HyperSamplingConfig(method="independent-fd", steps=steps, ddim_eta=0, **base))
    torch.testing.assert_close(torch.stack(zero), torch.stack(baseline))
    assert all(not step["guided"] for step in zero_stats["steps"])


def test_hyper_config_rejects_unsupported_steps_and_settings():
    with pytest.raises(ValueError):
        HyperSamplingConfig(steps=9)
    with pytest.raises(ValueError):
        HyperSamplingConfig(ddim_eta=1.1)
    with pytest.raises(ValueError):
        HyperSamplingConfig(guidance_eval="predicted-clean")
    with pytest.raises(ValueError):
        HyperSamplingConfig(unet_batch_size=0)
    with pytest.raises(ValueError, match="zero probe variance"):
        HyperSamplingConfig(guidance_steps=(8,))


@pytest.mark.parametrize("backbone,size", [("hyper-sdxl", 1024), ("hyper-sd15", 512)])
def test_cli_hyper_defaults_and_rejects_flow_probe(monkeypatch, capsys, backbone, size):
    import json
    import sys
    from fd_sgs.cli import main

    argv = ["fd-sgs", "--prompt", "test", "--backbone", backbone,
            "--reward", "brightness", "--dry-run"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    config = json.loads(capsys.readouterr().out)["configs"][0]
    assert config["steps"] == 8
    assert config["height"] == config["width"] == size
    assert config["cfg_scale"] == 5.0
    assert config["guidance_eval"] == "final-rollout"
    monkeypatch.setattr(sys, "argv", argv + ["--twin-sampler", "flow-grpo"])
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("backbone,steps", [("hyper-sdxl", 8), ("hyper-sdxl", 12),
                                            ("hyper-sd15", 8), ("hyper-sd15", 12)])
def test_pipeline_loader_selects_matching_lora(monkeypatch, backbone, steps):
    import argparse
    import importlib.util
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    from fd_sgs.cli import _load_pipeline
    from fd_sgs.hyper_sd import HYPER_MODELS, HYPER_LORAS

    pipe = ToyPipe(backbone == "hyper-sdxl")
    pipe.vae.enable_slicing = lambda: None
    pipe.vae.enable_tiling = lambda: None
    pipe.load_lora_weights = lambda repo, weight_name: calls.append((repo, weight_name))
    pipe.fuse_lora = lambda lora_scale: calls.append(("fused", lora_scale))
    pipe.to = lambda device: pipe
    calls = []
    model_class = StableDiffusionXLPipeline if backbone == "hyper-sdxl" else StableDiffusionPipeline
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(model_class, "from_pretrained", lambda model, **kw: (
        calls.append((model, kw)) or pipe))
    args = argparse.Namespace(backbone=backbone, steps=steps, model=HYPER_MODELS[backbone], dtype="float16",
                              hyper_lora_repo="ByteDance/Hyper-SD", lora_scale=1.0,
                              offload=False, device="cpu")
    actual = _load_pipeline(args)
    assert actual is pipe
    assert calls[0][0] == HYPER_MODELS[backbone]
    assert calls[0][1]["variant"] == "fp16"
    assert calls[1] == ("ByteDance/Hyper-SD", HYPER_LORAS[(backbone, steps)])
    assert calls[2] == ("fused", 1.0)
    assert actual.scheduler.config.timestep_spacing == "trailing"
