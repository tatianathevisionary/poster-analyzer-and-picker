"""Metric parsing, scoring, and ranking.

Parses the raw feature vector produced by :mod:`poster_analyzer.features`
into named metrics, then computes a weighted, min-max normalized score.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

# Metrics used for scoring, in the order weights are applied.
SCORING_METRICS: tuple[str, ...] = (
    "color_variation_rgb",
    "color_variation_hsv",
    "brightness",
    "lab_a_mean",
    "texture_contrast",
    "texture_homogeneity",
    "edge_density",
    "vertical_symmetry",
    "horizontal_symmetry",
)

# Weights for the final weighted score (sum to 1.0).
SCORING_WEIGHTS: Dict[str, float] = {
    "color_variation_rgb": 0.15,   # Color diversity (RGB)
    "color_variation_hsv": 0.15,   # Color diversity (HSV / perceptual)
    "brightness": 0.10,            # Overall brightness (LAB L mean)
    "lab_a_mean": 0.10,            # LAB a-channel mean (green<->red axis)
    "texture_contrast": 0.15,      # Texture distinctness
    "texture_homogeneity": 0.10,   # Texture smoothness
    "edge_density": 0.15,          # Visual complexity
    "vertical_symmetry": 0.05,     # Composition balance (asymmetry score)
    "horizontal_symmetry": 0.05,   # Composition balance (asymmetry score)
}


def parse_metrics(features: np.ndarray) -> Dict[str, float]:
    """Parse a raw feature vector into named scalar metrics.

    See :mod:`poster_analyzer.features` for the feature-vector layout.

    Note: ``lab_a_mean`` reads ``color_moments[3]`` which is the mean of the
    LAB **a** channel (green<->red axis), not saturation.
    """
    color_hist = features[:192]          # RGB + HSV histograms
    color_moments = features[192:201]    # LAB color moments
    texture_features = features[201:-7]  # Combined texture features
    composition_features = features[-7:]  # Composition features

    return {
        "color_variation_rgb": float(np.std(color_hist[:96])),
        "color_variation_hsv": float(np.std(color_hist[96:192])),
        "brightness": float(color_moments[0]),   # LAB L channel mean
        "lab_a_mean": float(color_moments[3]),    # LAB a channel mean
        "texture_contrast": float(np.mean(texture_features[:4])),    # GLCM contrast
        "texture_homogeneity": float(np.mean(texture_features[8:12])),  # GLCM homogeneity
        "edge_density": float(composition_features[-1]),
        "vertical_symmetry": float(composition_features[-3]),
        "horizontal_symmetry": float(composition_features[-2]),
    }


def calculate_poster_score(
    metrics: Dict[str, float], all_metrics: List[Dict[str, float]]
) -> float:
    """Compute a weighted, min-max normalized score for one poster.

    Each metric is normalized to ``[0, 1]`` against the min/max observed
    across ``all_metrics``, then combined using :data:`SCORING_WEIGHTS`.
    """
    metric_ranges = {
        key: {
            "min": min(poster[key] for poster in all_metrics),
            "max": max(poster[key] for poster in all_metrics),
        }
        for key in SCORING_METRICS
    }

    score = 0.0
    for key in SCORING_METRICS:
        min_val = metric_ranges[key]["min"]
        max_val = metric_ranges[key]["max"]
        normalized = (metrics[key] - min_val) / (max_val - min_val + 1e-10)
        score += normalized * SCORING_WEIGHTS[key]

    return score


def clean_prompt_from_filename(filename: str) -> str:
    """Extract and clean the Midjourney prompt from a filename."""
    import re

    prompt = os.path.splitext(filename)[0]
    prompt = re.sub(r"[\d.]+$", "", prompt)        # trailing numbers/dots
    prompt = re.sub(r"\s*\(\d+\)\s*$", "", prompt)  # (1), (2), ...
    prompt = re.sub(r"\s+$", "", prompt)            # trailing whitespace
    return prompt
