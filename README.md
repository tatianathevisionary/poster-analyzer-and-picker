# Poster Analyzer and Picker

A Python tool for analyzing and ranking poster images based on visual characteristics. It uses computer vision techniques to evaluate posters by color, texture, and composition, then ranks them and organizes them into top / remaining folders.

## Features

- **Comprehensive image analysis**:
  - Color features (RGB, HSV, LAB color spaces)
  - Texture analysis (Sobel gradients, GLCM, Local Binary Patterns)
  - Composition features (rule of thirds, symmetry, edge density)

- **Weighted ranking system**:
  - Color variation (30%)
  - Tone / color — LAB brightness + LAB a-channel mean (20%)
  - Texture quality — contrast + homogeneity (25%)
  - Composition — edge density / visual complexity (15%)
  - Balance — vertical + horizontal symmetry (10%)

- **Automatic organization**: copies the top-N posters into a `used posters/`
  folder and the rest into `unused posters/`. Originals are left untouched.

- **Self-documenting CSV reports** with per-poster metrics, rankings, the
  cleaned Midjourney prompt, and the analysis methodology.

## Installation

```bash
git clone https://github.com/tatianathevisionary/poster-analyzer-and-picker.git
cd poster-analyzer-and-picker

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e .            # installs the `poster-analyzer` command
# or, for dependencies only:
pip install -r requirements.txt
```

## Usage

Run against any directory of images. Nothing is hardcoded.

```bash
poster-analyzer --input-dir /path/to/posters --output-dir ./output --top-n 50 --no-display
```

Or run as a module without installing the console script:

```bash
python -m poster_analyzer --input-dir /path/to/posters --output-dir ./output
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | *(required)* | Directory of poster images (`.jpg`, `.jpeg`, `.png`, `.bmp`). |
| `--output-dir` | `./output` | Where reports and organized folders are written. |
| `--top-n` | `50` | Number of top-ranked posters placed in `used posters/`. |
| `--no-display` | off | Skip the matplotlib preview (batch / CI-safe). |
| `--verbose` | off | Enable debug-level logging. |

## Output

Written under `--output-dir`:

1. **Folders** (files are **copied**, originals untouched):
   - `used posters/` — the top-N ranked posters
   - `unused posters/` — the remaining posters
   - Filename collisions are resolved by appending `_1`, `_2`, ...

2. **CSV reports** (each prefixed with a methodology header):
   - `poster_analysis_TIMESTAMP.csv` — all posters, ranked
   - `used posters/top_<N>_posters_TIMESTAMP.csv`
   - `unused posters/remaining_posters_TIMESTAMP.csv`

3. **Summary**: `poster_organization_summary.txt` — every poster with its rank.

## Project structure

```
poster_analyzer/
  features.py    # color / texture / composition feature extraction
  scoring.py     # metric parsing, weighted normalized scoring
  analysis.py    # discover -> extract -> score -> rank pipeline
  io_utils.py    # image discovery, CSV reports, file organization, display
  cli.py         # argparse CLI / console entry point
tests/
  test_poster_analyzer.py
```

## Notes on metrics

- Symmetry metrics are **asymmetry scores**: the mean absolute difference
  between the image and its mirror. Lower means more symmetric (0 = perfect).
  `vertical_symmetry` mirrors top/bottom; `horizontal_symmetry` mirrors left/right.
- `lab_a_mean` is the mean of the LAB **a** channel (green↔red axis), not saturation.

## Development

```bash
pip install -e ".[test]"
pytest
```

Tests use synthetic numpy images — no real files required.

## License

MIT — see the LICENSE file.

## Acknowledgments

- OpenCV for image processing
- scikit-image for advanced texture/feature analysis
