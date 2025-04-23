# Poster Analyzer and Picker

A sophisticated Python-based tool for analyzing and ranking posters based on various visual characteristics. This tool uses computer vision and machine learning techniques to evaluate posters based on color, texture, and composition features.

## Features

- **Comprehensive Image Analysis**: Evaluates posters based on multiple visual characteristics:
  - Color features (RGB, HSV, LAB color spaces)
  - Texture analysis (Sobel gradients, GLCM, Local Binary Patterns)
  - Composition features (rule of thirds, symmetry, edge density)

- **Automated Ranking System**: Ranks posters based on a weighted scoring system that considers:
  - Color variation (30%)
  - Visual appeal (20%)
  - Texture quality (25%)
  - Composition (15%)
  - Balance (10%)

- **Smart Organization**: Automatically organizes posters into:
  - Top 50 posters (highest ranked)
  - Remaining posters

- **Detailed Reporting**: Generates comprehensive CSV reports including:
  - Detailed metrics for each poster
  - Ranking information
  - Original Midjourney prompts
  - Analysis methodology

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tatianathevisionary/poster-analyzer-and-picker.git
cd poster-analyzer-and-picker
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your posters in the `originals` directory within your project folder.

2. Run the analyzer:
```bash
python poster_analyzer.py
```

3. The script will:
   - Analyze all posters in the directory
   - Generate rankings
   - Create separate folders for top 50 and remaining posters
   - Generate detailed CSV reports for each category

## Output

The script generates the following outputs:

1. **Folder Organization**:
   - `used posters/`: Contains the top 50 ranked posters
   - `unused posters/`: Contains the remaining posters

2. **CSV Reports**:
   - `top_50_posters_TIMESTAMP.csv`: Detailed analysis of top 50 posters
   - `remaining_posters_TIMESTAMP.csv`: Analysis of remaining posters
   - Each CSV includes:
     - Poster metrics
     - Ranking information
     - Original prompts
     - Analysis methodology

3. **Summary File**:
   - `poster_organization_summary.txt`: Lists all posters with their rankings

## Analysis Methodology

The tool uses a sophisticated analysis pipeline:

1. **Color Analysis**:
   - RGB and HSV histograms (32 bins per channel)
   - LAB color space moments (mean, std, skewness)
   - Color variation in different color spaces

2. **Texture Analysis**:
   - Sobel gradient analysis
   - GLCM (Gray Level Co-occurrence Matrix)
   - Local Binary Patterns

3. **Composition Analysis**:
   - Rule of thirds
   - Vertical and horizontal symmetry
   - Edge density and visual complexity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for image processing capabilities
- scikit-learn for machine learning components
- scikit-image for advanced image analysis features 