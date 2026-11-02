"""Exercise report generation without downloading a dataset or pretrained weights."""
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from iad import experiments


def test_runner_writes_only_observed_metrics(tmp_path, monkeypatch):
    root = tmp_path/'data'/'bottle'
    rng = np.random.default_rng(10)
    for relative in [*[f'train/good/{i}.png' for i in range(6)], 'test/good/0.png', 'test/defect/0.png']:
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(path)
    mask_path = root/'ground_truth/defect/0_mask.png'
    mask_path.parent.mkdir(parents=True)
    mask = np.zeros((16,16), np.uint8); mask[4:8,4:8] = 255
    Image.fromarray(mask).save(mask_path)

    class FakeExtractor:
        def __init__(self, *args):
            pass

        def extract(self, path, transform='original', seed=42):
            value = int.from_bytes(hashlib.sha256(path.read_bytes()).digest()[:4], 'little')
            generator = np.random.default_rng(value)
            result = {'layer2': generator.normal(size=(4,4,8)).astype('f'),
                      'layer3': generator.normal(size=(2,2,12)).astype('f')}
            return {k: v + (0.1 if transform != 'original' else 0) for k,v in result.items()}

    monkeypatch.setattr(experiments, 'FeatureExtractor', FakeExtractor)
    cfg = dict(output=str(tmp_path/'results'), seed=42, dataset='mvtec', data_root=str(root.parent),
        category='bottle', shots=4, calibration_images=2, image_size=16, device='cpu', cache=str(tmp_path/'cache'),
        methods=['unified'], conditions=['original','shifted'], checkpoint_root=str(tmp_path/'checkpoints'),
        model=dict(selected=3, graph_samples=16, neighbors=3, max_bank=8, smooth_sigma=1))
    rows = experiments.run(cfg)
    assert len(rows) == 2
    frame = pd.read_csv(tmp_path/'results/paper_tables/4_unified.csv')
    assert set(frame[frame.status == 'measured'].method) == {'unified'}
    assert frame[frame.status == 'pending'].image_auroc.isna().all()
    assert (tmp_path/'results/manifest.json').exists()
    assert len(list((tmp_path/'results/paper_figures').glob('*.png'))) >= 4
