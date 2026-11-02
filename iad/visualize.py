"""Measured-result plots and stage/channel explanations."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from .features import photometric


def panels(path, mask, prediction, model, output, transform="original", seed=42):
    size = model.config.image_size
    image = Image.open(path).convert("RGB").resize((size, size))
    if transform == "shifted":
        image = photometric(image, seed, shifted=True)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.ravel()
    axes[0].imshow(image); axes[0].set_title("Input")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Ground truth")
    axes[2].imshow(prediction["map"], cmap="inferno"); axes[2].set_title("Anomaly map")
    axes[3].imshow(image); axes[3].imshow(prediction["map"], cmap="inferno", alpha=.5); axes[3].set_title("Overlay")
    axes[4].imshow(prediction["map"] > model.threshold, cmap="gray"); axes[4].set_title("Normal-calibrated mask")
    stage, explanation = next(iter(prediction["explanation"].items()))
    patch_id = int(np.argmax(explanation["distances"]))
    image_id, row, col = explanation["provenance"][patch_id]
    reference = Image.open(model.paths[image_id]).convert("RGB").resize((size, size))
    h, w = explanation["raw_map"].shape
    crop = reference.crop((int(col*size/w), int(row*size/h), int((col+1)*size/w), int((row+1)*size/h)))
    axes[5].imshow(crop); axes[5].set_title(f"Nearest normal grid cell ({stage})")
    values = explanation["squared_contributions"][patch_id]
    top = np.argsort(values)[-8:][::-1]
    axes[6].bar(range(len(top)), values[top]); axes[6].set_xticks(range(len(top)), explanation["channels"][top], rotation=45)
    axes[6].set_title(f"{stage}: squared-distance terms")
    axes[6].set_xlabel("Selected channel ID")
    axes[7].imshow(explanation["squared_contributions"][:, top[0]].reshape(h, w), cmap="inferno")
    axes[7].set_title(f"Channel {explanation['channels'][top[0]]} contribution")
    for i, ax in enumerate(axes):
        if i != 6:
            ax.axis("off")
    fig.suptitle("Grid-cell reference is not the full CNN receptive field; channel terms explain raw patch distance.")
    fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)
    if len(prediction["explanation"]) > 1:
        fig, axes = plt.subplots(1, len(model.state)+1, figsize=(12, 4))
        for ax, (name, detail) in zip(axes, prediction["explanation"].items()):
            ax.imshow(detail["normalized_map"], cmap="inferno")
            ax.set_title(f"{name}, weight={detail['weight']:.3f}"); ax.axis("off")
        axes[-1].imshow(prediction["map"], cmap="inferno"); axes[-1].set_title("Fused + smoothed"); axes[-1].axis("off")
        fig.tight_layout(); fig.savefig(Path(output).with_name(Path(output).stem + "_fusion.png"), dpi=140); plt.close(fig)


def comparison(frame, output):
    measured = frame[frame.status == "measured"]
    if measured.empty:
        return
    for metric in ["image_auroc", "pixel_auroc", "selected_features", "memory_bytes"]:
        pivot = measured.pivot(index="method", columns="condition", values=metric)
        ax = pivot.plot.bar(figsize=(12, 5))
        ax.set_ylabel(metric); ax.set_ylim(bottom=0)
        if "auroc" in metric:
            ax.set_ylim(0, 1)
        plt.tight_layout(); plt.savefig(Path(output)/(metric+".png"), dpi=150); plt.close()
