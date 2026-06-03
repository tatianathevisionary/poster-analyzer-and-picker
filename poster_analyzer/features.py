"""Image feature extraction.

Extracts color, texture, and composition features from poster images.

Feature vector layout (used by the parsing helpers in ``scoring.py``):

    index   0 : 192   RGB + HSV histograms (6 channels x 32 bins)
    index 192 : 201   LAB color moments (L/a/b x mean, std, skewness)
    index 201 :  -7   texture features (Sobel + GLCM + LBP)
    index  -7 :       composition features (4 rule-of-thirds points,
                        vertical_symmetry, horizontal_symmetry, edge_density)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

logger = logging.getLogger(__name__)

# Image is resized to this square size before feature extraction.
TARGET_SIZE = (224, 224)


def extract_color_features(img: np.ndarray) -> np.ndarray:
    """Extract color histogram and LAB moment features from an RGB image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # RGB histogram features (32 bins per channel).
    hist_r = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()

    # HSV histogram features.
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

    # Color moments (mean, std, skewness) for each channel in LAB space.
    moments: list[float] = []
    for channel in cv2.split(lab):
        mean = float(np.mean(channel))
        std = float(np.std(channel))
        skew = float(np.cbrt(np.mean(np.power(channel - mean, 3))))
        moments.extend([mean, std, skew])

    return np.concatenate([hist_r, hist_g, hist_b, hist_h, hist_s, hist_v, moments])


def extract_texture_features(gray: np.ndarray) -> np.ndarray:
    """Extract Sobel, GLCM, and Local Binary Pattern texture features."""
    # Sobel gradients.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # GLCM features over four orientations.
    glcm = graycomatrix(
        gray.astype("uint8"),
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    glcm_features: list[float] = []
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        glcm_features.extend(graycoprops(glcm, prop).flatten())

    # Local Binary Patterns.
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()

    return np.concatenate(
        [
            [float(np.mean(magnitude)), float(np.std(magnitude))],  # Sobel features
            glcm_features,  # GLCM features
            lbp_hist,  # LBP histogram
        ]
    )


def extract_composition_features(gray: np.ndarray) -> np.ndarray:
    """Extract rule-of-thirds, symmetry, and edge-density composition features.

    Symmetry metrics are *asymmetry* scores: they measure the mean absolute
    difference between the image and its mirror. LOWER values mean MORE
    symmetric (a perfectly symmetric image scores 0).

    - ``vertical_symmetry`` mirrors top<->bottom with ``np.flipud`` and so
      measures symmetry about the horizontal (vertical-axis) midline.
    - ``horizontal_symmetry`` mirrors left<->right with ``np.fliplr`` and so
      measures symmetry about the vertical midline.
    """
    h, w = gray.shape
    h1, h2 = h // 3, 2 * h // 3
    w1, w2 = w // 3, 2 * w // 3

    # Measure activity at the four rule-of-thirds intersection points.
    thirds_points: list[float] = []
    for hi in (h1, h2):
        for wi in (w1, w2):
            region = gray[hi - 5 : hi + 5, wi - 5 : wi + 5]
            thirds_points.append(float(np.mean(region)))

    gray_f = gray.astype(np.float64)
    # flipud = top/bottom mirror -> symmetry across the horizontal midline.
    vertical_symmetry = float(np.mean(np.abs(gray_f - np.flipud(gray_f))))
    # fliplr = left/right mirror -> symmetry across the vertical midline.
    horizontal_symmetry = float(np.mean(np.abs(gray_f - np.fliplr(gray_f))))

    # Edge density for visual complexity.
    edges = cv2.Canny(gray.astype("uint8"), 100, 200)
    edge_density = float(np.mean(edges > 0))

    return np.array(
        [*thirds_points, vertical_symmetry, horizontal_symmetry, edge_density]
    )


def extract_features_from_array(img_rgb: np.ndarray) -> np.ndarray:
    """Extract the full feature vector from an in-memory RGB image array."""
    img = cv2.resize(img_rgb, TARGET_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    color_features = extract_color_features(img)
    texture_features = extract_texture_features(gray)
    composition_features = extract_composition_features(gray)

    return np.concatenate([color_features, texture_features, composition_features])


def extract_features(image_path: str) -> Optional[np.ndarray]:
    """Extract the full feature vector from an image file.

    Returns ``None`` if the image cannot be loaded or processed.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Could not load image: %s", image_path)
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return extract_features_from_array(img)
    except Exception as exc:  # noqa: BLE001 - log and skip bad images
        logger.error("Error processing image %s: %s", image_path, exc)
        return None
