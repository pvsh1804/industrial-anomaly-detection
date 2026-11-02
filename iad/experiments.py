"""One-category, one-seed runner with resumable features and honest tables."""
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import platform
import random
import time
import numpy as np
import pandas as pd
import torch
import yaml
from .data import mvtec, visa, manifest_dataset, normal_split
from .features import FeatureExtractor
from .model import Detector, ModelConfig
from .metrics import evaluate
from .visualize import panels, comparison

METHODS = {
    "patchcore": {},
    "variance": {"variance": True},
    "laplacian": {"laplacian": True},
    "fewshot": {"variance": True, "laplacian": True},
    "consistency": {"consistency": True},
    "illumination": {"consistency": True, "correlation": True},
    "layer2": {"stages": "layer2"},
    "layer3": {"stages": "layer3"},
    "multiscale": {"stages": "multi"},
    "spec": {"stages": "multi", "spec": True},
    "adaptive": {"stages": "multi", "spec": True, "adaptive": True},
    "unified": {"stages": "multi", "variance": True, "laplacian": True, "consistency": True,
                "correlation": True, "spec": True, "adaptive": True},
}
TABLES = {
    "1_fewshot": ["patchcore", "variance", "laplacian", "fewshot"],
    "2_illumination": ["patchcore", "laplacian", "consistency", "illumination"],
    "3_multiscale": ["layer2", "layer3", "multiscale", "spec", "adaptive"],
    "4_unified": ["fewshot", "illumination", "adaptive", "unified"],
    "5_efficiency": list(METHODS),
}


def write_tables(records, output, conditions):
    rows = {(r["method"], r["condition"]): r for r in records}
    all_rows = [rows.get((m, c), dict(method=m, condition=c, status="pending", image_auroc=None, pixel_auroc=None, aupro=None, selected_features=None, memory_bytes=None, detector_seconds_per_image=None)) for m in METHODS for c in conditions]
    frame = pd.DataFrame(all_rows)
    (output/"paper_tables").mkdir(parents=True, exist_ok=True)
    for name, methods in TABLES.items():
        frame[frame.method.isin(methods)].to_csv(output/"paper_tables"/(name+".csv"), index=False)
    return frame


def run(cfg):
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output/"paper_figures").mkdir(exist_ok=True)
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    conditions = cfg.get("conditions", ["original", "shifted"])
    records = []
    write_tables(records, output, conditions)
    (output/"run_config.yaml").write_text(yaml.safe_dump(cfg))
    try:
        if cfg["dataset"] == "mvtec":
            train, test = mvtec(cfg["data_root"], cfg["category"])
        elif cfg["dataset"] == "visa":
            train, test = visa(cfg["data_root"], cfg["category"])
        elif cfg["dataset"] == "mvtec_ad2":
            train, test = manifest_dataset(cfg["data_root"], cfg["manifest"], cfg["category"])
        else:
            raise ValueError("Unsupported dataset")
        fit, calibration = normal_split(train, cfg["shots"], cfg["calibration_images"], seed)
        max_test = cfg.get("max_test_images")
        if max_test:
            # Deterministic per-subtype sampling for a speed-limited diagnostic.
            groups = sorted({s.image.parent.name for s in test})
            test = [s for group in groups for s in [x for x in test if x.image.parent.name == group][:max_test]]
        all_fit_hashes = {hashlib.sha256(s.image.read_bytes()).hexdigest() for s in fit+calibration}
        if any(hashlib.sha256(s.image.read_bytes()).hexdigest() in all_fit_hashes for s in test):
            raise ValueError("Duplicate image content crosses training and test splits")
        extractor = FeatureExtractor(cfg["image_size"], cfg["device"], cfg["cache"], cfg.get("threads", 2))
        print("Extracting normal features", flush=True)
        original = [extractor.extract(s.image) for s in fit]
        transformed = [extractor.extract(s.image, "fit_photo", seed+i) for i, s in enumerate(fit)]
        heldout = [extractor.extract(s.image) for s in calibration]
        test_features, feature_seconds = {}, {}
        for condition in conditions:
            print(f"Extracting {len(test)} test images: {condition}", flush=True)
            start = time.perf_counter()
            test_features[condition] = [extractor.extract(s.image, condition, seed+10000+i) for i, s in enumerate(test)]
            feature_seconds[condition] = (time.perf_counter()-start)/len(test)
        manifest = {"fit": [str(s.image) for s in fit], "calibration": [str(s.image) for s in calibration],
                    "test": [str(s.image) for s in test], "python": platform.python_version(), "torch": torch.__version__,
                    "device": cfg["device"], "fit_shots": len(fit), "extra_normal_calibration_images": len(calibration),
                    "image_sha256": {str(s.image): hashlib.sha256(s.image.read_bytes()).hexdigest() for s in fit+calibration+test},
                    "feature_seconds_per_image_including_cache_io": feature_seconds}
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
        masks = np.stack([s.load_mask(cfg["image_size"]) for s in test])
        for method in cfg["methods"]:
            print(f"Fitting {method}", flush=True)
            options = dict(cfg.get("model", {})) | METHODS[method]
            model = Detector(ModelConfig(**(options | dict(seed=seed, image_size=cfg["image_size"]))))
            start = time.perf_counter()
            model.fit(original, transformed, heldout, [s.image for s in fit])
            fit_seconds = time.perf_counter()-start
            model.save(Path(cfg.get("checkpoint_root", "checkpoints"))/output.name/method)
            (output/(method+"_model.json")).write_text(json.dumps(model.metadata(), indent=2))
            for condition in conditions:
                # Untimed warm-up, then detector-only timing with cached embeddings.
                model.predict(test_features[condition][0])
                start = time.perf_counter()
                predictions = [model.predict(f) for f in test_features[condition]]
                seconds = (time.perf_counter()-start)/len(test)
                metrics = evaluate([s.label for s in test], masks, [p["map"] for p in predictions],
                                   [p["score"] for p in predictions], model.threshold, cfg.get("max_fpr", .3))
                (output/(method+"_"+condition+"_metrics.json")).write_text(json.dumps(metrics, indent=2))
                metrics.pop("pro_curve")
                record = dict(method=method, condition=condition, status="measured", **metrics,
                              fit_seconds=fit_seconds, detector_seconds_per_image=seconds,
                              selected_features=model.metadata()["selected_features"], memory_bytes=model.metadata()["memory_bytes"],
                              bank_entries=model.metadata()["bank_entries"], fit_shots=len(fit),
                              calibration_images=len(calibration), test_images=len(test), seed=seed)
                records.append(record)
                frame = write_tables(records, output, conditions)
                print(json.dumps(record), flush=True)
                # One anomalous and one normal diagnostic; no selection by score.
                selected = [next((i for i, s in enumerate(test) if s.label == label), None) for label in [1, 0]]
                for i in selected:
                    if i is None:
                        continue
                    explanation = model.predict(test_features[condition][i], explain=True)
                    panels(test[i].image, masks[i], explanation, model,
                           output/"paper_figures"/f"{method}_{condition}_{i}.png", condition, seed+10000+i)
        comparison(frame, output/"paper_figures")
        note = ("Preliminary measurements only: one category, one seed. All fitting/selection uses normal training images.\n"
                f"Protocol: {len(fit)} fitting shots plus {len(calibration)} additional normal calibration images; not strict k-total.\n"
                "Baseline follows the manuscript's unweighted nearest-neighbor/max-smoothed-map formulation, not the original PatchCore reweighted image score.\n"
                "Inference timings are detector-only on cached embeddings; cache/extraction timing is separately recorded.\n"
                "AUPRO is a 200-threshold approximation with 8-connected regions and configured FPR limit.\n"
                "No full benchmark, multi-seed confidence intervals, VisA, or MVTec AD2 results are claimed.\n"
                "SPEC with the manuscript's identity spectral function equals centered Laplacian Score; it is not an independent ranking novelty.\n"
                "Stage-wise squared-distance terms are exact; linear c_j/A allocations can reconstruct the fused/smoothed distance map.\n")
        (output/"RESEARCH_FINDINGS.md").write_text(note)
        return records
    except Exception as error:
        (output/"RESEARCH_FINDINGS.md").write_text(f"Execution incomplete: {type(error).__name__}: {error}\n"
            "Only rows marked measured are actual results. All other experiments remain pending.\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--shots", type=int, choices=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.shots:
        config["shots"] = args.shots
        config["output"] += f"_k{args.shots}"
    run(config)
