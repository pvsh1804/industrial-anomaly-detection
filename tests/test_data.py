import numpy as np
import pytest
from PIL import Image
from iad.data import mvtec, normal_split


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def test_mvtec_masks_and_disjoint_split(tmp_path):
    for i in range(8):
        write(tmp_path/f'bottle/train/good/{i}.png', i)
    write(tmp_path/'bottle/test/good/0.png', 9)
    write(tmp_path/'bottle/test/crack/0.png', 10)
    write(tmp_path/'bottle/ground_truth/crack/0_mask.png', 255)
    train, test = mvtec(tmp_path, 'bottle')
    fit, holdout = normal_split(train, 4, 2, 42)
    assert not ({s.image for s in fit} & {s.image for s in holdout})
    assert len(fit) == 4
    assert sum(s.load_mask(8).sum() for s in test) == 64
    with pytest.raises(ValueError):
        normal_split(train, 8, 2, 42)


def test_missing_defect_mask_fails(tmp_path):
    write(tmp_path/'bottle/train/good/0.png', 5)
    write(tmp_path/'bottle/test/crack/0.png', 10)
    with pytest.raises(FileNotFoundError):
        mvtec(tmp_path, 'bottle')


def test_mask_geometry_mismatch_fails(tmp_path):
    write(tmp_path/'bottle/train/good/0.png', 5)
    write(tmp_path/'bottle/test/crack/0.png', 10)
    path = tmp_path/'bottle/ground_truth/crack/0_mask.png'
    path.parent.mkdir(parents=True)
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match='dimensions'):
        mvtec(tmp_path, 'bottle')
