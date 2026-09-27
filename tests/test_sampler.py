from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline

from fd_sgs.sampler import SamplingConfig, sample


class ToyTransformer:
    in_channels = 3
    dtype = torch.float32

    def __init__(self):
        self.calls = []

    def __call__(self, states, times, embeds, return_dict=False):
        assert len(states) == len(times) == len(embeds)
        assert states[0].shape == (3, 1, 2, 2)
        self.calls.append((len(states), times.clone()))
        return ([.2 * x + .1 * t for x, t in zip(states, times)],)


class ToyVAE:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=.5, shift_factor=.1)

    def decode(self, z, return_dict=False):
        return (z,)


class ToyPipe:
    """Tiny deterministic components with the REAL scheduler and native pipeline loop."""
    _execution_device = torch.device("cpu")
    vae_scale_factor = 8
    guidance_scale = ZImagePipeline.guidance_scale
    do_classifier_free_guidance = ZImagePipeline.do_classifier_free_guidance
    interrupt = ZImagePipeline.interrupt
    prepare_latents = ZImagePipeline.prepare_latents

    def __init__(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3)
        self.transformer = ToyTransformer()
        self.vae = ToyVAE()
        # Keep floating-point outputs to compare without quantization hiding bugs.
        self.image_processor = SimpleNamespace(postprocess=lambda x, output_type: list(x.unbind()))
        self.freed = False

    def encode_prompt(self, **kwargs):
        return [torch.ones(2, 4)], None

    def maybe_free_model_hooks(self):
        self.freed = True

    def progress_bar(self, total):
        return nullcontext(SimpleNamespace(update=lambda: None))


def reward(images, prompt):
    return [float(x.mean()) for x in images]


def config(**kwargs):
    return SamplingConfig(height=16, width=16, particles=2, **kwargs)


def test_unguided_matches_native_nine_step_pipeline():
    pipe = ToyPipe()
    images, stats = sample(pipe, "test", reward, config(method="unguided"))
    native = ToyPipe()
    expected = ZImagePipeline.__call__(native, prompt="test", height=16, width=16,
                                      num_images_per_prompt=2, num_inference_steps=9,
                                      guidance_scale=0., generator=torch.Generator().manual_seed(42)).images
    torch.testing.assert_close(torch.stack(images), torch.stack(expected))
    assert len(native.transformer.calls) == 9
    assert len(pipe.transformer.calls) == stats["sequential_nfe"] == 8
    assert stats["denoiser_batch_elements"] == 16
    assert stats["reward_image_evaluations"] == 2


@pytest.mark.parametrize("method", ["fd-sgs", "independent-fd"])
@pytest.mark.parametrize("twin_sampler", ["edm", "flow-grpo"])
@pytest.mark.parametrize("probes", [1, 3])
def test_guidance_schedule_budget_and_reproducibility(method, twin_sampler, probes):
    pipe = ToyPipe()
    cfg = config(method=method, twin_sampler=twin_sampler, probes=probes)
    images, stats = sample(pipe, "test", reward, cfg)
    repeated, repeated_stats = sample(pipe, "test", reward, cfg)
    torch.testing.assert_close(torch.stack(images), torch.stack(repeated))
    assert stats["final_rewards"] == repeated_stats["final_rewards"]
    assert stats["sequential_nfe"] == 8
    assert stats["denoiser_batch_elements"] == 8 * 2 * (1 + probes)
    assert stats["twin_trajectories"] == 2 * probes
    assert stats["twin_sampler"] == twin_sampler
    assert stats["reward_calls"] == 4
    assert stats["reward_image_evaluations"] == 3 * 2 * (1 + probes) + 2
    assert [x["step"] for x in stats["steps"] if x["guided"]] == [3, 5, 6]
    for step in stats["steps"]:
        if step["guided"]:
            assert max(step["correction_ratios"]) <= .025001
    assert pipe.freed


@pytest.mark.parametrize("twin_sampler", ["edm", "flow-grpo"])
def test_zero_rho_preserves_anchor_and_single_particle_matches_independent(twin_sampler):
    baseline, _ = sample(ToyPipe(), "test", reward, config(method="unguided"))
    disabled, _ = sample(ToyPipe(), "test", reward, config(rho=0, twin_sampler=twin_sampler, probes=3))
    torch.testing.assert_close(torch.stack(disabled), torch.stack(baseline))
    cfg = replace(config(), particles=1, twin_sampler=twin_sampler, probes=3)
    stein, _ = sample(ToyPipe(), "test", reward, cfg)
    independent, _ = sample(ToyPipe(), "test", reward, replace(cfg, method="independent-fd"))
    torch.testing.assert_close(torch.stack(stein), torch.stack(independent))


@pytest.mark.parametrize("twin_sampler", ["edm", "flow-grpo"])
def test_guidance_changes_output_and_zero_exploration_has_no_fd_signal(twin_sampler):
    baseline, _ = sample(ToyPipe(), "test", reward, config(method="unguided"))
    guided, _ = sample(ToyPipe(), "test", reward, config(twin_sampler=twin_sampler, probes=3))
    assert not torch.allclose(torch.stack(baseline), torch.stack(guided))
    no_probe, _ = sample(ToyPipe(), "test", reward, config(exploration=0, noise_level=0,
                                                        repulsion=0, twin_sampler=twin_sampler, probes=3))
    torch.testing.assert_close(torch.stack(baseline), torch.stack(no_probe))


@pytest.mark.parametrize("bad", [[float("nan")]*4, [1.]])
def test_bad_reward_fails_and_releases_hooks(bad):
    pipe = ToyPipe()
    with pytest.raises(ValueError, match="finite scalar"):
        sample(pipe, "test", lambda images, prompt: bad, config())
    assert pipe.freed


@pytest.mark.parametrize("kwargs", [{"particles": 0}, {"rho": -1}, {"exploration": float("nan")},
                                   {"probes": 0}, {"probes": 1.5}, {"twin_sampler": "unknown"},
                                   {"noise_level": -1}, {"noise_level": float("inf")},
                                   {"height": 17}, {"steps": 9}, {"guidance_steps": (0, 3)}])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)


def test_multi_probe_pairing_and_averaging(monkeypatch):
    import fd_sgs.sampler as sampler
    original_fd = sampler.fd_direction
    observed = {}

    def record_pairs(anchor, twin, anchor_reward, twin_reward):
        # Probe-major batches must repeat the same K anchors for every probe.
        for i in range(1, 3):
            torch.testing.assert_close(anchor[:2], anchor[2*i:2*i+2])
            torch.testing.assert_close(anchor_reward[:2], anchor_reward[2*i:2*i+2])
        result = original_fd(anchor, twin, anchor_reward, twin_reward)
        observed["expected"] = torch.stack([result[:2], result[2:4], result[4:6]]).mean(0)
        return result

    def record_stein(states, directions, repulsion):
        torch.testing.assert_close(directions, observed["expected"])
        observed["calls"] = observed.get("calls", 0) + 1
        return directions

    monkeypatch.setattr(sampler, "fd_direction", record_pairs)
    monkeypatch.setattr(sampler, "stein_field", record_stein)
    sample(ToyPipe(), "test", reward, config(probes=3, twin_sampler="flow-grpo"))
    assert observed["calls"] == 3


@pytest.mark.parametrize("wandb_mode", ["disabled", "offline"])
def test_cli_comparison_writes_images_and_metrics(monkeypatch, tmp_path, wandb_mode):
    import json
    import sys
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor
    from fd_sgs.cli import main

    pipe = ToyPipe()
    pipe.vae.enable_slicing = lambda: None
    pipe.vae.enable_tiling = lambda: None
    pipe.to = lambda device: pipe
    pipe.image_processor = VaeImageProcessor()
    monkeypatch.setattr(ZImagePipeline, "from_pretrained", lambda *a, **kw: pipe)
    monkeypatch.setattr(sys, "argv", ["fd-sgs", "--prompt", "test", "--method", "compare",
                        "--particles", "2", "--probes", "3", "--twin-sampler", "flow-grpo",
                        "--noise-level", "0.03", "--reward", "brightness", "--device", "cpu",
                        "--dtype", "float32", "--height", "16", "--width", "16",
                        "--output", str(tmp_path), "--wandb-mode", wandb_mode, "--wandb-previews"])
    main()
    summary = json.loads((tmp_path / "comparison.json").read_text())
    assert set(summary) == {"unguided", "independent-fd", "fd-sgs"}
    for method in summary:
        target = tmp_path / method
        with Image.open(target / "best.png") as image:
            image.verify()
        assert len(list(target.glob("particle_*.png"))) == 2
        data = json.loads((target / "metrics.json").read_text())
        assert data["config"]["noise_level"] == .03
        assert data["metrics"]["sequential_nfe"] == 8
        assert data["metrics"]["denoiser_batch_elements"] == (16 if method == "unguided" else 64)
    if wandb_mode == "offline":
        assert list((tmp_path / "wandb").glob("offline-run-*/*.wandb"))
        assert list((tmp_path / "wandb").rglob("*.table.json"))
        assert list((tmp_path / "wandb").rglob("*.png"))
    with pytest.raises(SystemExit):
        main()  # Existing results are protected from overwriting.


def test_step_callback_does_not_change_sampling_or_budget():
    traces = []
    previews = []
    def callback(trace, images):
        traces.append(trace)
        if images is not None:
            previews.append(images)
    cfg = config(probes=2)
    reference, baseline = sample(ToyPipe(), "test", reward, cfg)
    actual, stats = sample(ToyPipe(), "test", reward, cfg, on_step=callback)
    torch.testing.assert_close(torch.stack(actual), torch.stack(reference))
    assert len(traces) == 8
    assert len(previews) == 3
    assert all(len(batch) == 6 for batch in previews)
    for key in ("sequential_nfe", "reward_calls", "reward_image_evaluations", "denoiser_batch_elements"):
        assert stats[key] == baseline[key]


@pytest.mark.parametrize("wandb_mode", ["disabled", "offline"])
def test_batch_prompts_four_final_rewards_and_seed_pairing(monkeypatch, tmp_path, wandb_mode):
    import csv
    import json
    import sys
    from diffusers.image_processor import VaeImageProcessor
    from fd_sgs.cli import main
    from fd_sgs.rewards import RewardRegistry

    pipe = ToyPipe()
    pipe.vae.enable_slicing = lambda: None
    pipe.vae.enable_tiling = lambda: None
    pipe.to = lambda device: pipe
    pipe.image_processor = VaeImageProcessor()
    monkeypatch.setattr(ZImagePipeline, "from_pretrained", lambda *a, **kw: pipe)
    monkeypatch.setattr(RewardRegistry, "check_available", lambda *a, **kw: None)
    calls = []
    def fake_scorer(name):
        def score(images, prompt):
            calls.append((name, prompt, len(images)))
            return [float(image.convert("L").getpixel((0, 0))) / 255 for image in images]
        return score
    monkeypatch.setattr(RewardRegistry, "get", lambda self, name: fake_scorer(name))
    monkeypatch.setattr(sys, "argv", ["fd-sgs", "--prompt-dataset", "hpsv2", "--max-prompts", "2",
                        "--method", "compare", "--particles", "2", "--reward", "pickscore",
                        "--eval-rewards", "all", "--device", "cpu", "--dtype", "float32",
                        "--height", "16", "--width", "16", "--output", str(tmp_path),
                        "--wandb-mode", wandb_mode])
    main()
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert len(rows) == 12  # 2 prompts * 3 methods * 2 particles
    assert set(row["method"] for row in rows) == {"unguided", "independent-fd", "fd-sgs"}
    assert set(row["prompt_index"] for row in rows) == {"0", "1"}
    assert set(row["seed"] for row in rows) == {"42", "43"}
    assert set(row["prompt_id"] for row in rows) == {"0", "1"}
    assert all(set(("imagereward", "pickscore", "hpsv2", "clipscore")) <= row.keys() for row in rows)
    for index in (0, 1):
        assert {row["seed"] for row in rows if row["prompt_index"] == str(index)} == {str(42 + index)}
        for method in ("unguided", "independent-fd", "fd-sgs"):
            target = tmp_path / f"prompt_{index:05d}" / method
            data = json.loads((target / "metrics.json").read_text())
            assert len(data["metrics"]["evaluation_scores"]) == 4
            assert data["metrics"]["evaluation_reward_image_evaluations"] == 6
    summary = json.loads((tmp_path / "batch_summary.json").read_text())
    assert summary["methods"]["fd-sgs"]["prompts"] == 2
    assert len([x for x in calls if x[0] == "hpsv2"]) == 6
    if wandb_mode == "offline":
        assert len(list(tmp_path.glob("prompt_*/wandb/offline-run-*/*.wandb"))) == 2
        assert list(tmp_path.rglob("*.table.json"))


def test_dry_run_lists_dataset_without_models_or_overwriting(monkeypatch, tmp_path, capsys):
    import json
    import sys
    from fd_sgs.cli import main
    existing = tmp_path / "prompt_00000" / "fd-sgs"
    existing.mkdir(parents=True)
    (existing / "best.png").write_bytes(b"keep")
    monkeypatch.setattr(ZImagePipeline, "from_pretrained", lambda *a, **kw:
                        pytest.fail("dry run must not load the model"))
    monkeypatch.setattr(sys, "argv", ["fd-sgs", "--prompt-dataset", "image-reward",
                        "--max-prompts", "1", "--eval-rewards", "all", "--dry-run",
                        "--output", str(tmp_path)])
    main()
    planned = json.loads(capsys.readouterr().out)
    assert planned["prompts"][0]["id"] == "005695-0057"
    assert len(planned["scorers"]) == 4
    assert (existing / "best.png").read_bytes() == b"keep"
