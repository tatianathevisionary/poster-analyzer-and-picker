"""I/O helpers: image discovery, CSV reporting, file organization, display."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Shared methodology block, written into every CSV report so the file is
# self-documenting. Previously this text was duplicated three times.
METHODOLOGY_LINES: List[str] = [
    "Poster Analysis Parameters:",
    "1. Color Features:",
    "   - RGB and HSV histograms (32 bins per channel)",
    "   - LAB color space moments (mean, std, skewness)",
    "   - Color variation in different color spaces",
    "",
    "2. Texture Features:",
    "   - Sobel gradient analysis (edges and details)",
    "   - GLCM (contrast, dissimilarity, homogeneity, energy, correlation)",
    "   - Local Binary Patterns (local texture patterns)",
    "",
    "3. Composition Features:",
    "   - Rule of thirds analysis",
    "   - Vertical and horizontal symmetry (asymmetry scores; lower = more symmetric)",
    "   - Edge density (visual complexity)",
    "",
    "4. Ranking Criteria (weighted):",
    "   - Color variation (30%): Diversity of colors in RGB and HSV spaces",
    "   - Tone/color (20%): LAB brightness and LAB a-channel mean",
    "   - Texture quality (25%): Contrast and homogeneity",
    "   - Composition (15%): Edge density and visual complexity",
    "   - Balance (10%): Vertical and horizontal symmetry",
    "",
    "5. Additional Information:",
    "   - midjourney_prompt: Original prompt used to generate the image",
    "   - Score: Higher scores indicate better performance across all metrics",
    "",
    "Data starts below:",
]


def methodology_dataframe() -> pd.DataFrame:
    """Return the shared methodology block as a single-column DataFrame."""
    return pd.DataFrame({"Analysis Methodology": METHODOLOGY_LINES})


def find_images(image_dir: str) -> List[str]:
    """Return sorted paths of all supported images directly under ``image_dir``."""
    paths = [
        str(p)
        for p in Path(image_dir).glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def write_report_csv(df: pd.DataFrame, output_file: str) -> None:
    """Write a methodology header followed by ``df`` to ``output_file``."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fh:
        methodology_dataframe().to_csv(fh, index=False)
        fh.write("\n")
        df.to_csv(fh, index=False)
    logger.info("Wrote report: %s", output_file)


def _resolve_collision(dest_path: str) -> str:
    """Return a non-colliding destination path by appending ``_1``, ``_2``, ..."""
    if not os.path.exists(dest_path):
        return dest_path
    root, ext = os.path.splitext(dest_path)
    n = 1
    candidate = f"{root}_{n}{ext}"
    while os.path.exists(candidate):
        n += 1
        candidate = f"{root}_{n}{ext}"
    return candidate


def organize_posters_by_rank(
    df: pd.DataFrame, output_dir: str, top_n: int
) -> Tuple[str, str]:
    """Copy posters into ``used``/``unused`` folders by rank and write CSVs.

    Files are COPIED (originals are left untouched). Returns the two CSV paths.
    """
    used_dir = os.path.join(output_dir, "used posters")
    unused_dir = os.path.join(output_dir, "unused posters")
    os.makedirs(used_dir, exist_ok=True)
    os.makedirs(unused_dir, exist_ok=True)

    df_used = df[df["rank"] <= top_n].copy()
    df_unused = df[df["rank"] > top_n].copy()

    copied: dict[str, list[tuple[str, int]]] = {"used": [], "unused": []}

    logger.info("Organizing posters into folders (top_n=%d)...", top_n)
    for _, row in df.iterrows():
        source_path = row["file_path"]
        filename = row["filename"]
        rank = int(row["rank"])
        try:
            if rank <= top_n:
                dest_path = _resolve_collision(os.path.join(used_dir, filename))
                copied["used"].append((filename, rank))
            else:
                dest_path = _resolve_collision(os.path.join(unused_dir, filename))
                copied["unused"].append((filename, rank))
            shutil.copy2(source_path, dest_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error copying file %s: %s", filename, exc)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    used_csv = os.path.join(used_dir, f"top_{top_n}_posters_{timestamp}.csv")
    unused_csv = os.path.join(unused_dir, f"remaining_posters_{timestamp}.csv")
    write_report_csv(df_used, used_csv)
    write_report_csv(df_unused, unused_csv)

    logger.info(
        "Poster organization complete: copied %d to 'used posters', %d to 'unused posters'.",
        len(copied["used"]),
        len(copied["unused"]),
    )

    summary_file = os.path.join(output_dir, "poster_organization_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write(f"Used Posters (Top {top_n}):\n")
        fh.write("-" * 50 + "\n")
        for filename, rank in sorted(copied["used"], key=lambda x: x[1]):
            fh.write(f"Rank {rank}: {filename}\n")
        fh.write("\nUnused Posters:\n")
        fh.write("-" * 50 + "\n")
        for filename, rank in sorted(copied["unused"], key=lambda x: x[1]):
            fh.write(f"Rank {rank}: {filename}\n")
    logger.info("Wrote summary: %s", summary_file)

    return used_csv, unused_csv


def display_images(image_paths: List[str], n_cols: int = 5) -> None:
    """Display images in a grid using matplotlib (imported lazily)."""
    import cv2
    import matplotlib.pyplot as plt

    n_images = len(image_paths)
    n_rows = (n_images + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))
    for i, path in enumerate(image_paths):
        plt.subplot(n_rows, n_cols, i + 1)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Poster {i + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
