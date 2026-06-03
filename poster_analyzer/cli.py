"""Command-line interface for the poster analyzer."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .analysis import analyze_all_posters
from .io_utils import display_images, organize_posters_by_rank

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poster-analyzer",
        description="Analyze and rank poster images by color, texture, and composition.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the poster images to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for CSV reports and organized poster folders (default: ./output).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top-ranked posters to place in 'used posters' (default: 50).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the matplotlib display of the top posters (batch/CI-safe).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    results_df = analyze_all_posters(args.input_dir, args.output_dir)
    if results_df is None:
        logger.error("No results produced. Check --input-dir: %s", args.input_dir)
        return 1

    total = len(results_df)
    logger.info("Analyzed and ranked all %d posters.", total)

    top = results_df.head(10)
    logger.info("Top %d ranked posters:", len(top))
    for _, row in top.iterrows():
        logger.info(
            "Rank %d/%d (score %.3f): %s",
            row["rank"], total, row["score"], row["midjourney_prompt"],
        )

    organize_posters_by_rank(results_df, args.output_dir, args.top_n)

    if not args.no_display:
        top_paths = results_df.head(min(20, total))["file_path"].tolist()
        try:
            display_images(top_paths)
        except Exception as exc:  # noqa: BLE001 - display is best-effort
            logger.warning("Could not display images: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
