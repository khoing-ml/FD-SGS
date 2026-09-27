"""Black-box reward adapters; no reward gradients or batch softmax."""

import importlib

import torch


REWARD_NAMES = ("imagereward", "pickscore", "hpsv2", "clipscore")


def normalize_reward(name):
    if ":" in name:
        return name  # Preserve case for custom Python imports.
    key = name.lower().replace("-", "").replace("_", "")
    key = {"pick": "pickscore", "clip": "clipscore", "hps": "hpsv2"}.get(key, key)
    if key not in (*REWARD_NAMES, "brightness"):
        raise ValueError(f"Unknown reward {name!r}; choose {', '.join(REWARD_NAMES)}, brightness, or module:function")
    return key


class PickScore:
    model_id = "yuvalkirstain/PickScore_v1"
    processor_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __init__(self, device="cpu", batch_size=4):
        from transformers import AutoModel, AutoProcessor
        self.device, self.batch_size = device, batch_size
        self.processor = AutoProcessor.from_pretrained(self.processor_id)
        self.model = AutoModel.from_pretrained(self.model_id).eval().to(device)
        self.model.requires_grad_(False)
        self._prompt, self._text = None, None

    @torch.inference_mode()
    def __call__(self, images, prompt):
        if prompt != self._prompt:
            inputs = self.processor(text=[prompt], padding=True, truncation=True, max_length=77,
                                    return_tensors="pt").to(self.device)
            text = self.model.get_text_features(**inputs)
            self._text = torch.nn.functional.normalize(text, dim=-1)
            self._prompt = prompt
        scores = []
        for i in range(0, len(images), self.batch_size):
            inputs = self.processor(images=images[i:i+self.batch_size], return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
            # Cosine reward matches SGS's score scale. Official PickScore uses
            # the same ranking with a learned positive logit scale.
            scores.extend((features @ self._text.T).flatten().float().cpu().tolist())
        return scores


class CLIPScore(PickScore):
    """SGS-style CLIP ViT-L/14 cosine similarity, without 100x scaling."""
    model_id = processor_id = "openai/clip-vit-large-patch14"


class ImageReward:
    """Official ImageReward-v1.0 score, including its mean/std normalization."""
    def __init__(self, device="cpu", batch_size=4):
        # ImageReward 1.5 imports helpers from their old Transformers location.
        # Z-Image requires a newer Transformers, where they live in pytorch_utils.
        from transformers import modeling_utils, pytorch_utils
        for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
            if not hasattr(modeling_utils, name):
                setattr(modeling_utils, name, getattr(pytorch_utils, name))
        try:
            import ImageReward as rm
        except ImportError as exc:
            raise ImportError("ImageReward needs: pip install -r requirements-rewards.txt") from exc
        self.model = rm.load("ImageReward-v1.0", device=device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, images, prompt):
        # Official single-image API preserves input order and bounds memory.
        return [float(self.model.score(prompt, image)) for image in images]


class HPSv2:
    """HPS weights on the matching OpenCLIP ViT-H/14 architecture.

    Use the full checkpoint, not the base LAION weights. This avoids hpsv2's
    global device selection, repeated weight loads, and obsolete pytest pins.
    """
    def __init__(self, device="cpu", batch_size=4, version="v2.0"):
        if version not in {"v2.0", "v2.1"}:
            raise ValueError("HPS version must be v2.0 or v2.1")
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("HPSv2 needs: pip install -r requirements-rewards.txt") from exc
        from huggingface_hub import hf_hub_download
        self.device, self.batch_size = device, batch_size
        filename = "HPS_v2_compressed.pt" if version == "v2.0" else "HPS_v2.1_compressed.pt"
        checkpoint_path = hf_hub_download("xswu/HPSv2", filename)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision="fp32", device="cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        self.model.eval().to(device).requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self._prompt, self._text = None, None

    @torch.inference_mode()
    def __call__(self, images, prompt):
        if self._prompt != prompt:
            tokens = self.tokenizer([prompt]).to(self.device)
            self._text = self.model.encode_text(tokens, normalize=True)
            self._prompt = prompt
        scores = []
        for i in range(0, len(images), self.batch_size):
            pixels = torch.stack([self.preprocess(image.convert("RGB"))
                                  for image in images[i:i+self.batch_size]]).to(self.device)
            features = self.model.encode_image(pixels, normalize=True)
            scores.extend((features @ self._text.T).flatten().float().cpu().tolist())
        return scores


class Brightness:
    """Download-free plumbing check, NOT a perceptual-quality benchmark."""
    def __call__(self, images, prompt):
        from PIL import ImageStat
        return [ImageStat.Stat(image.convert("L")).mean[0] / 255 for image in images]


class RewardRegistry:
    """Lazy model cache shared across prompts and methods."""
    def __init__(self, device="cpu", batch_size=4, hps_version="v2.0"):
        if batch_size < 1:
            raise ValueError("Reward batch size must be positive")
        self.device, self.batch_size, self.hps_version = device, batch_size, hps_version
        self.cache = {}

    def check_available(self, names):
        for name in names:
            module = {"imagereward": "ImageReward", "hpsv2": "open_clip"}.get(name)
            if module and importlib.util.find_spec(module) is None:
                raise ImportError(f"{name} needs: pip install -r requirements-rewards.txt")

    def get(self, name):
        name = normalize_reward(name)
        if name not in self.cache:
            factories = {"pickscore": PickScore, "clipscore": CLIPScore, "imagereward": ImageReward}
            if name in factories:
                scorer = factories[name](self.device, self.batch_size)
            elif name == "hpsv2":
                scorer = HPSv2(self.device, self.batch_size, self.hps_version)
            elif name == "brightness":
                scorer = Brightness()
            else:
                module, function = name.rsplit(":", 1)
                scorer = getattr(importlib.import_module(module), function)
            self.cache[name] = scorer
        return self.cache[name]

    def score(self, name, images, prompt):
        values = torch.as_tensor(self.get(name)(images, prompt), dtype=torch.float32).flatten()
        if len(values) != len(images) or not torch.isfinite(values).all():
            raise ValueError(f"{name} must return one finite scalar per image")
        return values.tolist()

    def describe(self, names):
        metadata = {
            "pickscore": {"model": PickScore.model_id, "processor": PickScore.processor_id, "scale": "cosine"},
            "clipscore": {"model": CLIPScore.model_id, "processor": CLIPScore.processor_id, "scale": "cosine"},
            "imagereward": {"model": "ImageReward-v1.0", "scale": "official mean/std normalized score"},
            "hpsv2": {"model": "xswu/HPSv2", "version": self.hps_version, "scale": "cosine",
                       "backend": "open-clip-torch", "architecture": "ViT-H-14"},
            "brightness": {"scale": "mean grayscale / 255"},
        }
        return {name: metadata.get(name, {"callable": name}) for name in names}
