"""Tests for the poster analyzer using synthetic numpy images (no real files)."""

from __future__ import annotations

import numpy as np

from poster_analyzer.features import (
    extract_composition_features,
    extract_features_from_array,
)
from poster_analyzer.scoring import (
    SCORING_METRICS,
    calculate_poster_score,
    parse_metrics,
)


def _synthetic_image(seed: int = 0) -> np.ndarray:
    """Return a deterministic random RGB image as uint8."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)


def test_feature_vector_shape_and_parsing():
    """Feature extraction returns a fixed-length vector that parses to all metrics."""
    img = _synthetic_image(1)
    features = extract_features_from_array(img)

    # 192 histogram + 9 LAB moments + texture + 7 composition.
    assert features.ndim == 1
    assert features.shape[0] > 192 + 9 + 7
    assert np.all(np.isfinite(features))

    metrics = parse_metrics(features)
    for key in SCORING_METRICS:
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_symmetric_image_has_low_asymmetry():
    """A left/right-mirror-symmetric grayscale image scores ~0 horizontal asymmetry."""
    half = np.random.default_rng(2).integers(0, 256, size=(100, 50), dtype=np.uint8)
    gray = np.hstack([half, np.fliplr(half)]).astype(np.uint8)

    comp = extract_composition_features(gray)
    horizontal_symmetry = comp[-2]  # left/right mirror asymmetry score
    assert horizontal_symmetry < 1e-6  # perfectly symmetric -> ~0


def test_scoring_runs_and_normalizes():
    """Scoring runs across multiple posters and returns finite, normalized scores."""
    all_metrics = [parse_metrics(extract_features_from_array(_synthetic_image(s)))
                   for s in range(4)]

    scores = [calculate_poster_score(m, all_metrics) for m in all_metrics]
    assert len(scores) == 4
    for s in scores:
        assert np.isfinite(s)
        # Weighted sum of [0,1]-normalized metrics with weights summing to 1.
        assert -1e-9 <= s <= 1.0 + 1e-9
