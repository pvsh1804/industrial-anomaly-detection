"""Independent ablations and integrated normal-only multi-scale model."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import softmax
from .features import align_features, resize_map
from .selection import (knn_graph, consistency_graph, variance_channels, laplacian_scores,
                        spec_scores, ranked_channels, correlation_filter)
from .memory import greedy_coreset, nearest_neighbors, channel_contributions


@dataclass
class ModelConfig:
    stages: str = "combined"
    variance: bool = False
    laplacian: bool = False
    consistency: bool = False
    correlation: bool = False
    spec: bool = False
    adaptive: bool = False
    selected: int = 64
    graph_samples: int = 512
    neighbors: int = 10
    variance_quantile: float = 0.1
    correlation_threshold: float = 0.95
    pair_weight: float = 1.0
    coreset_ratio: float = 0.05
    max_bank: int = 256
    temperature: float = 0.1
    seed: int = 42
    image_size: int = 224
    smooth_sigma: float = 4.0
    backend: str = "numpy"


class Detector:
    def __init__(self, config):
        self.config = config
        self.state = {}
        self.fitted = False
        if config.stages not in {"combined", "layer2", "layer3", "multi"}:
            raise ValueError("Unknown stage configuration")
        if config.adaptive and not config.spec:
            raise ValueError("Adaptive weights require SPEC scores")
        if config.temperature <= 0 or config.selected < 1 or config.graph_samples < 2:
            raise ValueError("Invalid selector/fusion configuration")

    def stages(self, features):
        if self.config.stages == "combined":
            return align_features(features)
        if self.config.stages == "multi":
            return features
        return {self.config.stages: features[self.config.stages]}

    def fit(self, original, transformed, calibration, paths):
        """Features must originate exclusively from normal training images.

        Calibration images are separate from fitting images. Their count is
        reported separately from the k fitting shots (not strict k-total).
        """
        c = self.config
        if not original or not calibration or len(original) != len(paths):
            raise ValueError("Fit features, calibration features, and provenance are required")
        if c.consistency and (transformed is None or len(transformed) != len(original)):
            raise ValueError("Missing corresponding photometric fit features")
        original = [self.stages(f) for f in original]
        calibration = [self.stages(f) for f in calibration]
        transformed = [self.stages(f) for f in transformed] if transformed else None
        self.state = {}
        rng = np.random.default_rng(c.seed)
        for stage in original[0]:
            x = np.concatenate([f[stage].reshape(-1, f[stage].shape[-1]) for f in original])
            provenance = np.concatenate([np.column_stack([np.full(f[stage].shape[0]*f[stage].shape[1], i),
                np.indices(f[stage].shape[:2]).reshape(2, -1).T]) for i, f in enumerate(original)]).astype(int)
            sample_ids = rng.choice(len(x), min(c.graph_samples, len(x)), replace=False)
            selected = variance_channels(x, c.variance_quantile) if c.variance else np.arange(x.shape[1])
            sample = x[sample_ids][:, selected]
            ranking = None
            if c.laplacian or c.consistency:
                if c.consistency:
                    xt = np.concatenate([f[stage].reshape(-1, f[stage].shape[-1]) for f in transformed])
                    sample, graph = consistency_graph(sample, xt[sample_ids][:, selected], c.neighbors, c.pair_weight)
                else:
                    graph = knn_graph(sample, c.neighbors)
                ranking = laplacian_scores(sample, graph)
                # Unified mode first prunes to twice the target count, then SPEC.
                local = ranked_channels(ranking, c.selected * (2 if c.spec else 1))
                if c.correlation:
                    local = correlation_filter(sample, local, c.correlation_threshold)
                selected = selected[local]
            elif c.correlation:
                selected = selected[correlation_filter(sample, np.arange(len(selected)), c.correlation_threshold)]
            spectral_quality = 0.0
            if c.spec:
                sample = x[sample_ids][:, selected]
                ranking = spec_scores(sample, knn_graph(sample, c.neighbors))
                local = ranked_channels(ranking, c.selected)
                spectral_quality = float(np.mean(ranking[local]))
                selected = selected[local]
            bank, bank_ids = greedy_coreset(x[:, selected], c.coreset_ratio, c.max_bank, c.seed)
            holdout = np.concatenate([f[stage].reshape(-1, f[stage].shape[-1])[:, selected] for f in calibration])
            distances, _ = nearest_neighbors(holdout, bank, backend=c.backend)
            scale = float(np.mean(distances)) if c.stages == "multi" else 1.0
            if scale <= 1e-12:
                raise ValueError("Degenerate held-out normal distance; inspect calibration split")
            self.state[stage] = dict(channels=selected, bank=bank, provenance=provenance[bank_ids],
                                     scale=scale, quality=spectral_quality)
        qualities = [state["quality"] for state in self.state.values()]
        self.weights = softmax(-np.asarray(qualities)/c.temperature) if c.adaptive else np.ones(len(qualities))/len(qualities)
        self.paths = [str(p) for p in paths]
        self.fitted = True
        # Operating threshold is estimated exclusively on normal holdout maps.
        holdout_maps = [self._predict_stages(f)["map"] for f in calibration]
        self.threshold = float(np.quantile(np.stack(holdout_maps), 0.995))
        return self

    def _predict_stages(self, features, explain=False):
        if not self.fitted:
            raise RuntimeError("Fit the detector before inference")
        c = self.config
        output = np.zeros((c.image_size, c.image_size), dtype=np.float64)
        details = {}
        for weight, (stage, state) in zip(self.weights, self.state.items()):
            array = features[stage]
            x = array.reshape(-1, array.shape[-1])[:, state["channels"]]
            distances, matches = nearest_neighbors(x, state["bank"], backend=c.backend)
            raw_map = distances.reshape(array.shape[:2])
            normalized = resize_map(raw_map, c.image_size)/state["scale"]
            output += weight * normalized
            if explain:
                contributions = channel_contributions(x, state["bank"][matches])
                # Exact additive distance allocation: c_j / A, whose sum is A.
                allocation = contributions / np.maximum(distances[:, None], 1e-15)
                details[stage] = dict(squared_contributions=contributions,
                    distance_contributions=allocation, distances=distances, matches=matches,
                    raw_map=raw_map, normalized_map=normalized, channels=state["channels"],
                    provenance=state["provenance"][matches], weight=float(weight), scale=state["scale"])
        output = gaussian_filter(output, c.smooth_sigma)
        return {"map": output, "score": float(output.max()), "explanation": details}

    def predict(self, features, explain=False):
        return self._predict_stages(self.stages(features), explain)

    def metadata(self):
        return dict(config=asdict(self.config), weights=self.weights.tolist(), threshold=self.threshold,
                    selected_features=sum(len(s["channels"]) for s in self.state.values()),
                    bank_entries=sum(len(s["bank"]) for s in self.state.values()),
                    memory_bytes=sum(s["bank"].nbytes for s in self.state.values()),
                    stages={stage: dict(channels=s["channels"].tolist(), scale=s["scale"], quality=s["quality"])
                            for stage, s in self.state.items()})

    def save(self, directory):
        import json
        from pathlib import Path
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = self.metadata() | {"paths": self.paths}
        (directory/"model.json").write_text(json.dumps(metadata, indent=2))
        np.savez_compressed(directory/"memory.npz", **{f"{stage}_{key}": value
            for stage, state in self.state.items() for key, value in state.items()
            if key in {"bank", "provenance"}})

    @classmethod
    def load(cls, directory):
        import json
        from pathlib import Path
        directory = Path(directory)
        metadata = json.loads((directory/"model.json").read_text())
        model = cls(ModelConfig(**metadata["config"]))
        with np.load(directory/"memory.npz", allow_pickle=False) as arrays:
            for stage, info in metadata["stages"].items():
                model.state[stage] = dict(channels=np.asarray(info["channels"]), scale=info["scale"],
                    quality=info["quality"], bank=arrays[f"{stage}_bank"], provenance=arrays[f"{stage}_provenance"])
        model.weights = np.asarray(metadata["weights"])
        model.paths, model.threshold, model.fitted = metadata["paths"], metadata["threshold"], True
        return model
