import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.distance import cdist
from iad.selection import (knn_graph, laplacian_scores, spec_scores, consistency_graph,
                           correlation_filter, variance_channels)
from iad.memory import nearest_neighbors, channel_contributions, greedy_coreset
from iad.metrics import pro_curve, safe_auroc
from iad.model import Detector, ModelConfig
from iad.experiments import METHODS
from iad.features import resize_map, photometric
from scipy.ndimage import gaussian_filter
from PIL import Image


def test_graph_symmetry_duplicates_and_no_self():
    x = np.array([[0, 0], [0, 0], [1, 2], [4, 5.]])
    graph = knn_graph(x, 2)
    np.testing.assert_allclose(graph.toarray(), graph.T.toarray())
    assert np.all(graph.diagonal() == 0)
    assert np.all(np.asarray(graph.sum(axis=1)) > 0)


def test_laplacian_against_dense_equation_and_spec():
    x = np.random.default_rng(1).normal(size=(50, 8))
    x[:, -1] = 1
    graph = knn_graph(x, 4)
    degree = np.diag(np.asarray(graph.sum(axis=1)).ravel())
    laplacian = degree - graph.toarray()
    centered = x - (np.ones(len(x)) @ degree @ x)/(np.ones(len(x)) @ degree @ np.ones(len(x)))
    expected = np.array([v @ laplacian @ v / (v @ degree @ v) for v in centered[:, :-1].T])
    actual = laplacian_scores(x, graph)
    np.testing.assert_allclose(actual[:-1], expected, rtol=1e-10)
    np.testing.assert_allclose(spec_scores(x, graph), actual, rtol=1e-10)
    assert np.isinf(actual[-1])


def test_consistency_pair_penalty():
    x = np.array([[0, 0], [1, 1], [2, 2], [3, 3.]], float)
    changed = x.copy(); changed[:, 1] += 20
    both, graph = consistency_graph(x, changed, neighbors=2, pair_weight=100)
    scores = laplacian_scores(both, graph)
    assert scores[0] < scores[1]
    assert graph[0, 4] >= 100


def test_filters():
    x = np.arange(20.)
    data = np.stack([x, 2*x, -x, np.ones(20)], axis=1)
    assert 3 not in variance_channels(data, 0)
    np.testing.assert_array_equal(correlation_filter(data, np.arange(4)), [0])
    with pytest.raises(ValueError):
        variance_channels(np.ones((4, 3)))


def test_nn_chunks_against_brute_force_and_explanation():
    rng = np.random.default_rng(2)
    x, bank = rng.normal(size=(31, 7)).astype('f'), rng.normal(size=(23, 7)).astype('f')
    distance, indices = nearest_neighbors(x, bank, chunk_size=5, bank_chunk=4)
    expected = cdist(x, bank)
    np.testing.assert_array_equal(indices, expected.argmin(axis=1))
    np.testing.assert_allclose(distance, expected.min(axis=1))
    np.testing.assert_allclose(channel_contributions(x, bank[indices]).sum(axis=1), distance**2)


def test_coreset_unique_duplicate_rows_and_reproducible():
    x = np.ones((10, 4), dtype='f')
    _, ids = greedy_coreset(x, ratio=.5)
    assert len(np.unique(ids)) == 5
    np.testing.assert_array_equal(ids, greedy_coreset(x, ratio=.5)[1])


def test_metrics_perfect_inverted_and_missing_classes():
    masks = np.zeros((2, 6, 6), dtype=bool)
    masks[0, 2:4, 2:4] = True
    assert pro_curve(masks, masks.astype(float))[0] == pytest.approx(1)
    assert pro_curve(masks, 1-masks.astype(float))[0] == pytest.approx(0)
    assert pro_curve(np.zeros_like(masks), np.zeros_like(masks, float))[0] is None
    assert safe_auroc([0, 0], [1, 2]) is None
    assert safe_auroc([0, 1], [1, 2]) == 1


def features(rng):
    return {"layer2": rng.normal(size=(4, 4, 12)).astype('f'),
            "layer3": rng.normal(size=(2, 2, 16)).astype('f')}


@pytest.mark.parametrize('method', list(METHODS))
def test_model_variants_roundtrip_and_additive_explanation(method, tmp_path):
    rng = np.random.default_rng(4)
    fit = [features(rng) for _ in range(4)]
    changed = [{k: v+.05*rng.normal(size=v.shape) for k, v in f.items()} for f in fit]
    heldout = [features(rng)]
    model = Detector(ModelConfig(**METHODS[method], selected=4, graph_samples=16,
        neighbors=3, max_bank=8, coreset_ratio=.2, image_size=16, smooth_sigma=1))
    model.fit(fit, changed, heldout, ['a', 'b', 'c', 'd'])
    query = features(rng)
    result = model.predict(query, explain=True)
    reconstruction = np.zeros((16, 16))
    for detail in result['explanation'].values():
        np.testing.assert_allclose(detail['squared_contributions'].sum(axis=1), detail['distances']**2, atol=1e-10)
        allocation_map = detail['distance_contributions'].sum(axis=1).reshape(detail['raw_map'].shape)
        reconstruction += detail['weight'] * resize_map(allocation_map, 16)/detail['scale']
    np.testing.assert_allclose(gaussian_filter(reconstruction, 1), result['map'], rtol=1e-6)
    assert np.isfinite(result['map']).all()
    model.save(tmp_path)
    restored = Detector.load(tmp_path)
    np.testing.assert_allclose(restored.predict(query)['map'], result['map'])
    np.testing.assert_allclose(model.weights.sum(), 1)


def test_photo_preserves_geometry():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[5:10, 10:15] = 200
    changed = np.asarray(photometric(Image.fromarray(image), 3))
    assert changed.shape == image.shape
    assert changed[7, 12].mean() > changed[0, 0].mean()


def test_calibration_is_required():
    with pytest.raises(ValueError, match='calibration'):
        Detector(ModelConfig()).fit([], None, [], [])


def test_variance_correlation_preserves_original_channel_ids():
    rng = np.random.default_rng(7)
    fit = [features(rng) for _ in range(2)]
    for feature in fit:
        feature['layer2'][..., 0] = 0
    model = Detector(ModelConfig(stages='layer2', variance=True, correlation=True,
        graph_samples=16, max_bank=4, image_size=16))
    model.fit(fit, None, [features(rng)], ['a', 'b'])
    assert 0 not in model.state['layer2']['channels']
