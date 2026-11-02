"""End-to-end wiring check on synthetic textures; NOT research evidence."""
from pathlib import Path
import json
import platform
import torch
import tempfile
import time
import numpy as np
from PIL import Image
from iad.features import FeatureExtractor
from iad.model import Detector, ModelConfig
from iad.experiments import METHODS
from iad.visualize import panels

output = Path('results/validation')
output.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
report = {'data': 'synthetic procedural textures; not MVTec and not paper evidence', 'models': {}, 'python': platform.python_version(), 'torch': torch.__version__, 'weights': 'Wide_ResNet50_2_Weights.IMAGENET1K_V1'}
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    yy, xx = np.mgrid[:128, :128]
    paths = []
    for i in range(7):
        texture = np.clip(120 + 15*np.sin(xx/4) + rng.normal(0, 2, (128, 128)), 0, 255).astype('uint8')
        image = np.repeat(texture[..., None], 3, axis=-1)
        if i == 6:
            image[45:60, 40:85] = 20
        path = root/f'{i}.png'
        Image.fromarray(image).save(path); paths.append(path)
    extractor = FeatureExtractor(size=128)
    original = [extractor.extract(p) for p in paths[:4]]
    changed = [extractor.extract(p, 'fit_photo', 42+i) for i,p in enumerate(paths[:4])]
    calibration = [extractor.extract(p) for p in paths[4:6]]
    start = time.perf_counter()
    query = extractor.extract(paths[6])
    report['uncached_query_feature_seconds'] = time.perf_counter()-start
    # Verify cache replay exactly, including photometric key separation.
    cached = extractor.extract(paths[0])
    for stage in original[0]:
        np.testing.assert_array_equal(cached[stage], original[0][stage])
    mask = np.zeros((128,128), bool); mask[45:60,40:85] = True
    for name in ['patchcore', 'fewshot', 'illumination', 'adaptive', 'unified']:
        model = Detector(ModelConfig(**METHODS[name], image_size=128, selected=32,
                         graph_samples=128, neighbors=5, max_bank=64, coreset_ratio=.05))
        model.fit(original, changed, calibration, paths[:4])
        start = time.perf_counter()
        prediction = model.predict(query, explain=True)
        report['models'][name] = {'finite_score': bool(np.isfinite(prediction['score'])),
            'selected_features': model.metadata()['selected_features'], 'memory_bytes': model.metadata()['memory_bytes'],
            'prediction_with_explanation_seconds': time.perf_counter()-start}
        assert report['models'][name]['finite_score']
        for detail in prediction['explanation'].values():
            np.testing.assert_allclose(detail['squared_contributions'].sum(axis=1), detail['distances']**2)
        panels(paths[6], mask, prediction, model, output/(name+'_synthetic_diagnostic.png'))
        model.save(root/name)
        restored = Detector.load(root/name)
        np.testing.assert_allclose(restored.predict(query)['map'], prediction['map'])
        print(name, report['models'][name], flush=True)
(output/'smoke.json').write_text(json.dumps(report, indent=2))
print('All five models passed pretrained-backbone synthetic smoke checks.')
