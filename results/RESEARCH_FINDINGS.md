# Execution status

Implementation and local validation completed; actual MVTec AD measurements are pending.

- 26 correctness and report-integration tests passed in 17.07 seconds on Python 3.11.17, PyTorch 2.5.1+cpu. Results are recorded in validation/pytest.xml and validation/environment.json.
- All five primary models completed an end-to-end synthetic-texture check using downloaded ImageNet-pretrained WideResNet-50-2 weights. Frozen extraction, caching, fitting, inference, channel decomposition and checkpoint round trips worked. See `validation/smoke.json`; these are software checks, not industrial-dataset evidence.
- Attempted `configs/quick.yaml` execution correctly stopped because no MVTec images exist in this workspace. No research AUROC/AUPRO values were created.
- The current official MVTec AD webpage uses a download form. Two previously distributed direct bottle URLs returned HTTP 404. A current official download URL or the bottle archive is needed to run the real experiment.
- Tables 1–5 mark every real-data experiment pending. No synthetic values are substituted.

## Methodological points for the manuscript

1. Equation (4)'s SPEC variant equals degree-centered Laplacian Score on the same graph. Do not claim they are mathematically independent rankings.
2. The planned quick run uses k=4 fitting images plus four separate normal calibration images. Report the extra normal-data budget.
3. The manuscript baseline implements equation (3), not the original PatchCore image-score reweighting.
4. Squared channel differences explain each raw stage distance exactly. Their c_j/A allocation can reconstruct the fused/smoothed distance map, but the fused score is not itself a sum of raw squared channel differences.
5. Local AUPRO is a 200-threshold approximation; validate publication numbers with the official dataset evaluator.
6. VisA and AD2 adapters/configurations exist, but have not been executed on those datasets. The AD2 adapter requires a user-prepared public-validation manifest.
7. CPU checks do not verify GPU numerical reproducibility or deployment latency. Report cache-based detector timing separately from extraction.

No conference-level accuracy, novelty, speedup, or illumination-robustness claim is established yet.
