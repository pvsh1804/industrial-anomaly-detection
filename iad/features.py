"""Frozen WRN50-2 features with content-addressed, transform-aware caching."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.nn import functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models.feature_extraction import create_feature_extractor


def photometric(image, seed, shifted=False):
    rng = np.random.default_rng(seed)
    bounds = (0.55, 1.45) if shifted else (0.8, 1.2)
    # Shifted evaluation uses disjoint gamma/gain ranges, same for every method.
    gamma = rng.choice([0.5, 1.6]) if shifted else rng.uniform(0.8, 1.2)
    gain = rng.choice([0.6, 1.4]) if shifted else rng.uniform(*bounds)
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(*bounds)))
    arr = np.asarray(image, dtype=np.float32) / 255
    yy, xx = np.mgrid[-1:1:complex(arr.shape[0]), -1:1:complex(arr.shape[1])]
    angle = rng.uniform(0, 2*np.pi)
    amplitude = 0.35 if shifted else rng.uniform(0.05, 0.2)
    shading = 1 + amplitude*(np.cos(angle)*xx + np.sin(angle)*yy)/np.sqrt(2)
    arr = np.clip((arr ** gamma) * gain * shading[..., None], 0, 1)
    return Image.fromarray(np.rint(arr*255).astype(np.uint8))


class FeatureExtractor:
    def __init__(self, size=224, device="cpu", cache="cache", threads=2):
        torch.set_num_threads(threads)
        self.size, self.device = size, torch.device(device)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = wide_resnet50_2(weights=self.weights)
        self.net = create_feature_extractor(backbone, {"layer2": "layer2", "layer3": "layer3"}).eval().to(self.device)
        self.net.requires_grad_(False)

    @torch.inference_mode()
    def extract(self, path, transform="original", seed=42):
        raw = Path(path).read_bytes()
        metadata = dict(schema=1, size=self.size, weights=str(self.weights), transform=transform, seed=seed,
                        torch=torch.__version__, pooling=3, alignment="bilinear_align_corners_false")
        key = hashlib.sha256(raw + json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        target = self.cache / (key + ".npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                return {stage: saved[stage] for stage in ["layer2", "layer3"]}
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        if transform != "original":
            if transform not in {"fit_photo", "shifted"}:
                raise ValueError("Unknown photometric transform")
            image = photometric(image, seed, shifted=transform == "shifted")
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()/255
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        maps = self.net(((tensor - mean)/std)[None].to(self.device))
        arrays = {stage: F.avg_pool2d(value, 3, 1, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                  for stage, value in maps.items()}
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(target)
        return arrays


def align_features(features):
    a, b = features["layer2"], features["layer3"]
    tensor = torch.from_numpy(b).permute(2, 0, 1)[None]
    resized = F.interpolate(tensor, size=a.shape[:2], mode="bilinear", align_corners=False)
    return {"combined": np.concatenate([a, resized[0].permute(1, 2, 0).numpy()], axis=-1)}


def resize_map(array, size):
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
