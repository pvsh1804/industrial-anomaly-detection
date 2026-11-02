"""Run a saved detector on one image; launched from repository root."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from iad.features import FeatureExtractor
from iad.model import Detector

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--output", default="results/inference")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
model = Detector.load(args.model)
extractor = FeatureExtractor(model.config.image_size, args.device)
result = model.predict(extractor.extract(args.image), explain=True)
output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
plt.imsave(output/"heatmap.png", result["map"], cmap="inferno")
np.save(output/"anomaly_map.npy", result["map"])
np.save(output/"predicted_mask.npy", result["map"] > model.threshold)
for stage, detail in result["explanation"].items():
    np.savez_compressed(output/(stage+"_explanation.npz"), **detail)
print({"image_score": result["score"], "normal_pixel_threshold": model.threshold, "output": str(output)})
