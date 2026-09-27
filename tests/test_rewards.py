import importlib.util
from PIL import Image
import pytest
import torch

from fd_sgs import rewards


def test_reward_registry_caches_and_checks_scores(monkeypatch):
    instances = []
    class FakeScorer:
        model_id = processor_id = "fake"
        def __init__(self, device, batch_size):
            instances.append((device, batch_size))
        def __call__(self, images, prompt):
            return [float(i) for i in range(len(images))]
    monkeypatch.setattr(rewards, "CLIPScore", FakeScorer)
    registry = rewards.RewardRegistry("cpu", 2)
    images = [Image.new("RGB", (4, 4)) for _ in range(3)]
    assert registry.score("clip_score", images, "cat") == [0., 1., 2.]
    assert registry.score("clipscore", images, "dog") == [0., 1., 2.]
    assert instances == [("cpu", 2)]
    assert registry.describe(["hpsv2"])["hpsv2"]["version"] == "v2.0"


@pytest.mark.parametrize("name", ["unknown", "hps-v1"])
def test_unsupported_reward_fails_early(name):
    with pytest.raises(ValueError, match="Unknown reward"):
        rewards.normalize_reward(name)


def test_reward_shape_and_nan_are_rejected():
    registry = rewards.RewardRegistry()
    images = [Image.new("RGB", (4, 4)) for _ in range(2)]
    registry.cache["pickscore"] = lambda images, prompt: [1.]
    with pytest.raises(ValueError, match="finite scalar"):
        registry.score("pickscore", images, "cat")
    registry.cache["pickscore"] = lambda images, prompt: [float("nan"), 1.]
    with pytest.raises(ValueError, match="finite scalar"):
        registry.score("pickscore", images, "cat")


@pytest.mark.skipif(importlib.util.find_spec("ImageReward") is None, reason="optional reward extra not installed")
def test_imagereward_import_shim_and_score_order(monkeypatch):
    from transformers import modeling_utils
    from transformers import pytorch_utils
    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(modeling_utils, name):
            monkeypatch.setattr(modeling_utils, name, getattr(pytorch_utils, name), raising=False)
    import ImageReward as rm
    class FakeModel:
        def eval(self): return self
        def requires_grad_(self, flag): return self
        def score(self, prompt, image): return image.getpixel((0, 0))[0] / 255
    monkeypatch.setattr(rm, "load", lambda *a, **kw: FakeModel())
    scorer = rewards.ImageReward("cpu")
    images = [Image.new("RGB", (2, 2), (0, 0, 0)), Image.new("RGB", (2, 2), (255, 0, 0))]
    assert scorer(images, "red") == [0., 1.]


@pytest.mark.skipif(importlib.util.find_spec("open_clip") is None, reason="optional reward extra not installed")
def test_hps_checkpoint_and_batched_score_contract(monkeypatch, tmp_path):
    import open_clip
    import huggingface_hub
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("value", torch.tensor([1.]))
        def encode_text(self, tokens, normalize=True):
            return torch.tensor([[1., 0.]])
        def encode_image(self, pixels, normalize=True):
            val = pixels.mean(dim=(1, 2, 3))
            output = torch.stack([val, 1-val], dim=-1)
            return torch.nn.functional.normalize(output, dim=-1)
    weights = tmp_path / "hps.pt"
    torch.save({"state_dict": FakeModel().state_dict()}, weights)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, filename: str(weights))
    monkeypatch.setattr(open_clip, "create_model_and_transforms", lambda *a, **kw:
                        (FakeModel(), None, lambda image: torch.full((3, 2, 2), image.getpixel((0, 0))[0] / 255)))
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda *a: lambda prompts: torch.ones((1, 2), dtype=torch.long))
    scorer = rewards.HPSv2("cpu", batch_size=1)
    images = [Image.new("RGB", (2, 2), (0, 0, 0)), Image.new("RGB", (2, 2), (255, 0, 0))]
    assert scorer(images, "white") == [0., 1.]
