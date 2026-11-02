"""Validated MVTec AD, VisA official split CSV, and AD2 manifest adapters."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    image: Path
    label: int
    mask: Path | None = None

    def load_mask(self, size):
        if self.mask is None:
            if self.label:
                raise ValueError(f"Anomalous sample has no public mask: {self.image}")
            return np.zeros((size, size), dtype=bool)
        return np.asarray(Image.open(self.mask).convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0


def validate(samples):
    if not samples:
        raise FileNotFoundError("No dataset images found")
    for sample in samples:
        if not sample.image.is_file() or (sample.mask is not None and not sample.mask.is_file()):
            raise FileNotFoundError(f"Missing image or mask: {sample}")
        with Image.open(sample.image) as image:
            shape = image.size
            image.verify()
        if sample.mask is not None:
            with Image.open(sample.mask) as mask:
                if mask.size != shape:
                    raise ValueError(f"Image/mask dimensions differ: {sample.image}")
                mask.verify()
    return samples


def mvtec(root, category):
    root = Path(root)/category
    train = [Sample(p, 0) for p in sorted((root/"train"/"good").glob("*.png"))]
    test = []
    for p in sorted((root/"test").glob("*/*.png")):
        label = int(p.parent.name != "good")
        mask = root/"ground_truth"/p.parent.name/(p.stem + "_mask.png") if label else None
        test.append(Sample(p, label, mask))
    return validate(train), validate(test)


def visa(root, category):
    root = Path(root)
    train, test = [], []
    with (root/"split_csv"/"1cls.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["object"] != category:
                continue
            label = int(row["label"].strip().lower() not in {"normal", "0", "good"})
            sample = Sample(root/row["image"], label, root/row["mask"] if label else None)
            if row["split"] == "train":
                if label:
                    raise ValueError("Only normal images can enter the training split")
                train.append(sample)
            elif row["split"] == "test":
                test.append(sample)
    return validate(train), validate(test)


def manifest_dataset(root, manifest, category):
    """AD2 explicit CSV: category,split,image,label,mask,lighting.

    Use public validation masks or official server evaluation for hidden tests.
    Hidden test labels/masks are deliberately not guessed.
    """
    root = Path(root)
    train, test = [], []
    with Path(manifest).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["category"] != category:
                continue
            sample = Sample(root/row["image"], int(row["label"]), root/row["mask"] if row.get("mask") else None)
            if row["split"] == "train":
                if sample.label:
                    raise ValueError("Non-normal training record")
                train.append(sample)
            elif row["split"] in {"validation", "test_public"}:
                test.append(sample)
    return validate(train), validate(test)


def normal_split(train, shots, calibration_images, seed):
    if any(s.label for s in train) or shots < 1 or calibration_images < 1:
        raise ValueError("Fit/calibration require normal images and positive counts")
    if shots + calibration_images > len(train):
        raise ValueError("Not enough disjoint normal fit and calibration images")
    ids = np.random.default_rng(seed).permutation(len(train))
    fit = [train[i] for i in ids[:shots]]
    calibration = [train[i] for i in ids[shots:shots+calibration_images]]
    # Exact image duplicates must not make held-out calibration falsely easy.
    fit_hashes = {hashlib.sha256(s.image.read_bytes()).digest() for s in fit}
    if any(hashlib.sha256(s.image.read_bytes()).digest() in fit_hashes for s in calibration):
        raise ValueError("Duplicate image content crosses fit/calibration split")
    return fit, calibration
