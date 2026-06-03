"""Poster Analyzer and Picker — analyze and rank poster images.

A small package that scores poster images by color, texture, and composition
features, ranks them, and organizes them into top / remaining folders.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .analysis import analyze_all_posters
from .features import (
    extract_color_features,
    extract_composition_features,
    extract_features,
    extract_features_from_array,
    extract_texture_features,
)
from .scoring import calculate_poster_score, parse_metrics

__all__ = [
    "__version__",
    "analyze_all_posters",
    "extract_features",
    "extract_features_from_array",
    "extract_color_features",
    "extract_texture_features",
    "extract_composition_features",
    "parse_metrics",
    "calculate_poster_score",
]
