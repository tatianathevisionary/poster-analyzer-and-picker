"""Top-level analysis pipeline: discover images, extract, score, rank."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from .features import extract_features
from .io_utils import find_images, write_report_csv
from .scoring import calculate_poster_score, clean_prompt_from_filename, parse_metrics

logger = logging.getLogger(__name__)


def analyze_all_posters(image_dir: str, output_dir: str) -> Optional[pd.DataFrame]:
    """Analyze every image in ``image_dir`` and return a ranked DataFrame.

    Writes a timestamped ``poster_analysis_*.csv`` into ``output_dir``.
    Returns ``None`` if no images were found.
    """
    image_paths = find_images(image_dir)
    if not image_paths:
        logger.warning("No images found in: %s", image_dir)
        return None

    logger.info("Found %d images to process.", len(image_paths))

    metrics_list: list[dict] = []
    for i, path in enumerate(image_paths, 1):
        logger.info("Processing image %d/%d", i, len(image_paths))
        features = extract_features(path)
        if features is None:
            continue
        filename = os.path.basename(path)
        metrics = parse_metrics(features)
        metrics_list.append(
            {
                "file_path": path,
                "filename": filename,
                "midjourney_prompt": clean_prompt_from_filename(filename),
                **metrics,
            }
        )

    if not metrics_list:
        logger.warning("No images could be processed successfully.")
        return None

    logger.info("Calculating rankings...")
    df = pd.DataFrame(metrics_list)
    df["score"] = [
        calculate_poster_score(row.to_dict(), metrics_list)
        for _, row in df.iterrows()
    ]
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("rank").reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"poster_analysis_{timestamp}.csv")
    write_report_csv(df, output_file)

    return df
