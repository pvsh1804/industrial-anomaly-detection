"""Image/pixel AUROC and region-balanced PRO; no test-tuned thresholds."""
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score


def safe_auroc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def pro_curve(masks, maps, max_fpr=0.3, steps=200):
    masks, maps = np.asarray(masks, bool), np.asarray(maps, float)
    if masks.shape != maps.shape or not np.isfinite(maps).all() or not 0 < max_fpr <= 1 or steps < 2:
        raise ValueError("Invalid PRO inputs")
    negatives = ~masks
    if negatives.sum() == 0:
        return None, [], []
    regions = []
    for index, mask in enumerate(masks):
        components, count = label(mask, structure=np.ones((3, 3)))
        for region in range(1, count + 1):
            regions.append((index, components == region))
    if not regions:
        return None, [], []
    thresholds = np.r_[np.inf, np.linspace(maps.max(), maps.min(), steps), -np.inf]
    fprs, pros = [], []
    for threshold in thresholds:
        prediction = maps > threshold
        fprs.append(float(prediction[negatives].mean()))
        pros.append(float(np.mean([prediction[i][region].mean() for i, region in regions])))
    # Preserve vertical steps at duplicate FPRs: collapsing them biases AUPRO,
    # e.g. a perfect detector must retain both (0,0) and (0,1).
    fprs, pros = np.asarray(fprs), np.asarray(pros)
    right = np.searchsorted(fprs, max_fpr, side="right")
    x, y = fprs[:right], pros[:right]
    if x[-1] < max_fpr:
        end_y = np.interp(max_fpr, fprs[right-1:right+1], pros[right-1:right+1])
        x, y = np.r_[x, max_fpr], np.r_[y, end_y]
    auc = float(np.trapezoid(y, x)/max_fpr)
    return auc, x.tolist(), y.tolist()


def evaluate(labels, masks, maps, scores, threshold, max_fpr=0.3, steps=200):
    masks, maps = np.asarray(masks, bool), np.asarray(maps, float)
    auc, fpr, pro = pro_curve(masks, maps, max_fpr, steps)
    components = []
    for mask, prediction in zip(masks, maps > threshold):
        ids, count = label(mask, structure=np.ones((3, 3)))
        components.extend([float(prediction[ids == i].mean()) for i in range(1, count+1)])
    return dict(image_auroc=safe_auroc(labels, scores), pixel_auroc=safe_auroc(masks.ravel(), maps.ravel()),
                aupro=auc, pro_at_normal_threshold=float(np.mean(components)) if components else None,
                pro_max_fpr=max_fpr, pro_threshold_steps=steps, pro_curve={"fpr": fpr, "pro": pro})
