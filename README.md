# Industrial surface anomaly detection

Normal-only research implementation accompanying **An Efficient and Explainable Unsupervised Framework for Industrial Surface Defect Detection Using Multi-Scale Feature Learning**. Implements the supplied manuscript's Sections III–IV. No backbone training is required.

## Status

Implemented: frozen WideResNet-50-2, pooled/aligned patch descriptors, greedy projected coreset, chunked exact nearest-neighbor matching, all three feature-selection solutions, unified model, inference, metrics, figures, and paper tables. Real MVTec results are **pending dataset availability**. Synthetic validation images are explicitly labeled and must not be used as research evidence.

## Install and run (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pytest -q
```

On Colab, retain its installed GPU-compatible PyTorch/torchvision, install `-e '.[test]'`, and change `device: cuda` in the YAML. See [the notebook](notebooks/quickstart.ipynb).

Obtain the bottle category from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). Its current page uses a download form; previously published direct bottle links returned HTTP 404 during this implementation. Use a current official archive URL:

```bash
python -m scripts.download_mvtec --url 'OFFICIAL_DOWNLOAD_URL'
# Or extract your archive so data/mvtec_ad/bottle/train/good exists.
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m iad.experiments --config configs/quick.yaml
```

The quick configuration runs the manuscript baseline, few-shot selection, illumination selection, multi-scale adaptive fusion, and unified model with k=4, seed=42. It evaluates original and controlled lighting-shift conditions. Full test split is the default; `max_test_images` optionally limits images **per subtype**, and that becomes a diagnostic subset rather than a benchmark result.

```bash
# All requested ablations (same features and split, cached automatically)
python -m iad.experiments --config configs/ablations.yaml
# Other supported shot counts: 1, 2, 4, 8, 16
python -m iad.experiments --config configs/quick.yaml --shots 8
# Single-image inference after a successful fit
python -m scripts.infer --model checkpoints/bottle_k4_seed42/unified \
  --image data/mvtec_ad/bottle/test/broken_large/000.png
# No-dataset wiring validation with actual pretrained backbone
python -m scripts.smoke
```

Results are saved under the configured output directory, including `paper_tables/`, `paper_figures/`, `RESEARCH_FINDINGS.md`, per-model metadata, metrics, run configuration, and image-hash/split manifests. Model banks are in ignored `checkpoints/`. Dataset files, weights, and feature caches are never committed.

## Methods

| Configuration | Normal-only feature selection | Representation |
|---|---|---|
| `patchcore` | None | Aligned layer2 + layer3 |
| `fewshot` | Variance quantile → Laplacian Score | Aligned layer2 + layer3 |
| `illumination` | kNN + photometric-pair Laplacian Score → Pearson filter | Aligned layer2 + layer3 |
| `adaptive` | SPEC per stage | Separate banks, normal-calibrated maps, softmax weights |
| `unified` | Variance → consistency Laplacian shortlist → Pearson → SPEC | Separate banks + adaptive fusion |

Individual switches are `ModelConfig` fields. `METHODS` in `iad/experiments.py` defines the named ablations. Unified mode retains up to twice the target channel count before the final SPEC ranking. All graph construction and channel selection use fitting images only. Photometric transformations preserve geometry. Calibration uses only normal training images, disjoint by image and exact image hash from fitting and testing.

## Mathematical and reporting limits

- The baseline follows the supplied manuscript's equation (3): unweighted Euclidean nearest-neighbor patch distance and maximum of the smoothed map. It is a **PatchCore-style manuscript baseline**, not an exact reproduction of the original implementation's image-score reweighting. Do not label its metrics as reproduced official PatchCore results.
- SPEC equation (4), with the identity spectral function and trivial-component removal, is algebraically equivalent to degree-centered Laplacian Score on the same graph. Its independent implementation is checked numerically. Multi-scale graphs differ; SPEC is not claimed as a separate new ranking function.
- The default protocol uses **4 fit images + 4 extra normal calibration images**. Report both; this is not a strict four-total-normal-image experiment. Calibration normalizes stage maps and chooses a 99.5th-percentile normal-pixel operating threshold. Strict k-total experiments need a separately specified calibration policy, particularly k=1.
- AUPRO uses 8-connected regions and 200 thresholds with interpolation at the configured false-positive-rate limit (0.30 for AD, 0.05 default for AD2). This local approximation is not a substitute for an official benchmark evaluator/server.
- Each raw stage distance satisfies `A² = sum(c_j)`, where `c_j = (x_j - m_j)²`. The fused/smoothed map is not a sum of these squared terms. `c_j/A` allocations sum to the stage distance; applying the same scale, fusion, interpolation and smoothing reconstructs the displayed distance map. The identity is covered by tests.
- Reference crops show matched spatial grid cells, not complete CNN receptive fields. Combined channel IDs 0–511 refer to layer2; 512–1535 refer to aligned layer3.
- Detector timing is measured on cached embeddings and is labeled accordingly. Feature extraction/cache time is reported separately. This is not end-to-end deployment latency.
- One category/seed cannot establish conference-level gains or robustness. Keep unexecuted cells pending; run full splits and multiple seeds before paper claims.

## Additional datasets

- [VisA archive](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar): `configs/visa.yaml` uses the official `split_csv/1cls.csv` normal-only split.
- [MVTec AD2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2): `configs/mvtec_ad2.yaml` uses an explicit CSV adapter with columns `category,split,image,label,mask,lighting`. Paths are relative to the dataset root; labels are 0/1; accepted evaluation splits are `validation` and `test_public`. It does not infer hidden test masks or claim official-server results. Create this manifest from your actual downloaded release before use.

Neither additional dataset is downloaded automatically. Adapters are ready for integration; real-release validation remains pending. MVTec AD is distributed under CC BY-NC-SA 4.0; obtain data from its official page and observe the dataset terms.
