import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
import re
import shutil

def extract_color_features(img):
    """Extract comprehensive color features from image."""
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # RGB histogram features
    hist_r = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()
    
    # HSV histogram features
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    
    # Color moments for each channel in LAB space
    moments = []
    for channel in cv2.split(lab):
        moments.extend([
            np.mean(channel),  # First moment - mean
            np.std(channel),   # Second moment - standard deviation
            np.cbrt(np.mean(np.power(channel - np.mean(channel), 3)))  # Third moment - skewness
        ])
    
    return np.concatenate([hist_r, hist_g, hist_b, hist_h, hist_s, hist_v, moments])

def extract_texture_features(gray):
    """Extract comprehensive texture features from grayscale image."""
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # GLCM features
    glcm = graycomatrix(gray.astype('uint8'), distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                       levels=256, symmetric=True, normed=True)
    glcm_features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        glcm_features.extend(graycoprops(glcm, prop).flatten())
    
    # Local Binary Patterns
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float) / sum(lbp_hist)
    
    return np.concatenate([
        [np.mean(magnitude), np.std(magnitude)],  # Sobel features
        glcm_features,  # GLCM features
        lbp_hist  # LBP histogram
    ])

def extract_composition_features(img, gray):
    """Extract composition-based features."""
    # Rule of thirds features
    h, w = gray.shape
    h1, h2 = h//3, 2*h//3
    w1, w2 = w//3, 2*w//3
    
    # Measure activity at intersection points
    thirds_points = []
    for hi in [h1, h2]:
        for wi in [w1, w2]:
            region = gray[hi-5:hi+5, wi-5:wi+5]
            thirds_points.append(np.mean(region))
    
    # Symmetry measures
    vertical_symmetry = np.mean(np.abs(gray - np.fliplr(gray)))
    horizontal_symmetry = np.mean(np.abs(gray - np.flipud(gray)))
    
    # Edge density for visual complexity
    edges = cv2.Canny(gray.astype('uint8'), 100, 200)
    edge_density = np.mean(edges > 0)
    
    return np.array([*thirds_points, vertical_symmetry, horizontal_symmetry, edge_density])

def extract_features(image_path):
    """Extract all features from an image."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        
        # Convert to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Extract all features
        color_features = extract_color_features(img)
        texture_features = extract_texture_features(gray)
        composition_features = extract_composition_features(img, gray)
        
        # Combine all features
        return np.concatenate([color_features, texture_features, composition_features])
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_images(image_paths):
    """Process all images and extract features."""
    features = []
    valid_paths = []
    
    for path in image_paths:
        feature = extract_features(path)
        if feature is not None:
            features.append(feature)
            valid_paths.append(path)
    
    return np.array(features), valid_paths

def select_representative_images(features, image_paths, n_clusters=20):
    """Select representative images using K-means clustering."""
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Select one image per cluster (closest to centroid)
    representative_indices = []
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) > 0:
            # Calculate distances to centroid
            distances = np.linalg.norm(features_scaled[cluster_indices] - kmeans.cluster_centers_[i], axis=1)
            # Select the closest image
            representative_idx = cluster_indices[np.argmin(distances)]
            representative_indices.append(representative_idx)
    
    return [image_paths[i] for i in representative_indices]

def display_images(image_paths, n_cols=5):
    """Display images using matplotlib."""
    n_images = len(image_paths)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 3*n_rows))
    for i, path in enumerate(image_paths):
        plt.subplot(n_rows, n_cols, i+1)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Poster {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_results_to_csv(representative_paths, features, cluster_labels):
    """Save the selected poster paths to a CSV file with timestamp and selection criteria."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/Users/tatiana/Downloads/Metaposters - Basketball Posters/top posters"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"selected_posters_{timestamp}.csv")
    
    # Calculate metrics for each selected image
    selection_metrics = []
    for path, feature in zip(representative_paths, features):
        # Parse different feature sections
        color_hist = feature[:192]  # RGB + HSV histograms
        color_moments = feature[192:201]  # LAB color moments
        texture_features = feature[201:-7]  # Combined texture features
        composition_features = feature[-7:]  # Composition features
        
        metrics = {
            'color_variation_rgb': np.std(color_hist[:96]),
            'color_variation_hsv': np.std(color_hist[96:192]),
            'brightness': color_moments[0],  # L channel mean
            'saturation': color_moments[3],  # a channel mean
            'texture_contrast': np.mean(texture_features[:4]),  # GLCM contrast
            'texture_homogeneity': np.mean(texture_features[8:12]),  # GLCM homogeneity
            'edge_density': composition_features[-1],
            'vertical_symmetry': composition_features[-3],
            'horizontal_symmetry': composition_features[-2]
        }
        selection_metrics.append(metrics)
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'poster_number': range(1, len(representative_paths) + 1),
        'file_path': representative_paths,
        'filename': [os.path.basename(path) for path in representative_paths],
        'selection_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **{k: [m[k] for m in selection_metrics] for k in selection_metrics[0].keys()}
    })
    
    # Add methodology information
    methodology_info = pd.DataFrame({
        'Selection Methodology': [
            'Image Analysis Parameters:',
            '1. Color Features:',
            '   - RGB and HSV histograms (32 bins per channel)',
            '   - LAB color space moments (mean, std, skewness)',
            '   - Color variation in different color spaces',
            '',
            '2. Texture Features:',
            '   - Sobel gradient analysis (edges and details)',
            '   - GLCM (contrast, dissimilarity, homogeneity, energy, correlation)',
            '   - Local Binary Patterns (local texture patterns)',
            '',
            '3. Composition Features:',
            '   - Rule of thirds analysis',
            '   - Vertical and horizontal symmetry',
            '   - Edge density (visual complexity)',
            '',
            '4. Selection Process:',
            '   - K-means clustering (k=20)',
            '   - StandardScaler for feature normalization',
            '   - Representative selection: closest to cluster centroid',
            '',
            'Metrics Explanation:',
            '- color_variation_rgb/hsv: Color diversity in different color spaces',
            '- brightness/saturation: Overall light and color intensity',
            '- texture_contrast/homogeneity: Pattern characteristics',
            '- edge_density: Amount of detail and complexity',
            '- vertical/horizontal_symmetry: Image balance measures',
            '',
            'Data starts below:'
        ]
    })
    
    # Save to CSV
    with open(output_file, 'w', encoding='utf-8') as f:
        methodology_info.to_csv(f, index=False)
        f.write('\n')
        df.to_csv(f, index=False)
    
    print(f"\nResults saved to: {output_file}")

def calculate_poster_score(metrics, all_metrics):
    """Calculate an overall score for a poster based on its metrics."""
    # Define which metrics to use for scoring
    scoring_metrics = [
        'color_variation_rgb',
        'color_variation_hsv',
        'brightness',
        'saturation',
        'texture_contrast',
        'texture_homogeneity',
        'edge_density',
        'vertical_symmetry',
        'horizontal_symmetry'
    ]
    
    # Get min and max values for each metric across all posters
    metric_ranges = {}
    for key in scoring_metrics:
        metric_ranges[key] = {
            'min': min(poster[key] for poster in all_metrics),
            'max': max(poster[key] for poster in all_metrics)
        }
    
    # Normalize each metric to 0-1 range
    normalized_metrics = {}
    for key in scoring_metrics:
        value = metrics[key]
        min_val = metric_ranges[key]['min']
        max_val = metric_ranges[key]['max']
        normalized_metrics[key] = (value - min_val) / (max_val - min_val + 1e-10)
    
    # Weighted scoring (adjust weights to prioritize different aspects)
    weights = {
        'color_variation_rgb': 0.15,  # Color diversity
        'color_variation_hsv': 0.15,  # Color diversity in perceptual space
        'brightness': 0.1,            # Overall brightness
        'saturation': 0.1,            # Color intensity
        'texture_contrast': 0.15,     # Texture distinctness
        'texture_homogeneity': 0.1,   # Texture smoothness
        'edge_density': 0.15,         # Visual complexity
        'vertical_symmetry': 0.05,    # Composition balance
        'horizontal_symmetry': 0.05    # Composition balance
    }
    
    # Calculate weighted score
    score = sum(normalized_metrics[key] * weight 
               for key, weight in weights.items())
    
    return score

def clean_prompt_from_filename(filename):
    """Extract and clean the Midjourney prompt from the filename."""
    # Remove file extension
    prompt = os.path.splitext(filename)[0]
    
    # Remove any trailing numbers (like '1.png1' or ' (1)')
    prompt = re.sub(r'[\d.]+$', '', prompt)  # Remove trailing numbers and dots
    prompt = re.sub(r'\s*\(\d+\)\s*$', '', prompt)  # Remove (1), (2), etc.
    prompt = re.sub(r'\s+$', '', prompt)  # Remove trailing whitespace
    
    return prompt

def analyze_all_posters(image_dir):
    """Analyze all posters and create a ranked list."""
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [str(p) for p in Path(image_dir).glob('*') 
                  if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        print("No images found in the specified directory.")
        return
    
    print(f"Found {len(image_paths)} images to process.")
    
    # Process all images
    metrics_list = []
    
    print("Analyzing images...")
    for i, path in enumerate(image_paths, 1):
        print(f"Processing image {i}/{len(image_paths)}", end='\r')
        
        features = extract_features(path)
        if features is not None:
            filename = os.path.basename(path)
            # Parse different feature sections
            color_hist = features[:192]  # RGB + HSV histograms
            color_moments = features[192:201]  # LAB color moments
            texture_features = features[201:-7]  # Combined texture features
            composition_features = features[-7:]  # Composition features
            
            metrics = {
                'file_path': path,
                'filename': filename,
                'midjourney_prompt': clean_prompt_from_filename(filename),
                'color_variation_rgb': np.std(color_hist[:96]),
                'color_variation_hsv': np.std(color_hist[96:192]),
                'brightness': color_moments[0],  # L channel mean
                'saturation': color_moments[3],  # a channel mean
                'texture_contrast': np.mean(texture_features[:4]),
                'texture_homogeneity': np.mean(texture_features[8:12]),
                'edge_density': composition_features[-1],
                'vertical_symmetry': composition_features[-3],
                'horizontal_symmetry': composition_features[-2]
            }
            
            metrics_list.append(metrics)
    
    print("\nCalculating rankings...")
    
    # Create DataFrame with all metrics
    df = pd.DataFrame(metrics_list)
    
    # Calculate scores and rankings
    scores = []
    for _, row in df.iterrows():
        score = calculate_poster_score(row.to_dict(), metrics_list)
        scores.append(score)
    
    df['score'] = scores
    # Rank all images from 1 to N
    df['rank'] = df['score'].rank(ascending=False, method='min').astype(int)
    
    # Sort by rank
    df = df.sort_values('rank')
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/Users/tatiana/Downloads/Metaposters - Basketball Posters/top posters"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"poster_analysis_{timestamp}.csv")
    
    # Add methodology information
    methodology_info = pd.DataFrame({
        'Analysis Methodology': [
            'Poster Analysis Parameters:',
            '1. Color Features:',
            '   - RGB and HSV histograms (32 bins per channel)',
            '   - LAB color space moments (mean, std, skewness)',
            '   - Color variation in different color spaces',
            '',
            '2. Texture Features:',
            '   - Sobel gradient analysis (edges and details)',
            '   - GLCM (contrast, dissimilarity, homogeneity, energy, correlation)',
            '   - Local Binary Patterns (local texture patterns)',
            '',
            '3. Composition Features:',
            '   - Rule of thirds analysis',
            '   - Vertical and horizontal symmetry',
            '   - Edge density (visual complexity)',
            '',
            '4. Ranking Criteria:',
            '   - Color variation (30%): Diversity of colors in RGB and HSV spaces',
            '   - Visual appeal (20%): Brightness and saturation',
            '   - Texture quality (25%): Contrast and homogeneity',
            '   - Composition (15%): Edge density and visual complexity',
            '   - Balance (10%): Vertical and horizontal symmetry',
            '',
            '5. Additional Information:',
            '   - midjourney_prompt: Original prompt used to generate the image',
            f'   - Rankings: All {len(df)} images are ranked from 1 (highest score) to {len(df)} (lowest score)',
            '   - Score: Higher scores indicate better performance across all metrics',
            '',
            'Data starts below:'
        ]
    })
    
    # Save to CSV with all images
    with open(output_file, 'w', encoding='utf-8') as f:
        methodology_info.to_csv(f, index=False)
        f.write('\n')
        df.to_csv(f, index=False)
    
    print(f"\nResults saved to: {output_file}")
    return df

def organize_posters_by_rank(df, base_dir):
    """Move posters to appropriate folders based on their ranking and create folder-specific CSVs."""
    # Create directories if they don't exist
    used_dir = os.path.join(base_dir, "used posters")
    unused_dir = os.path.join(base_dir, "unused posters")
    os.makedirs(used_dir, exist_ok=True)
    os.makedirs(unused_dir, exist_ok=True)
    
    # Keep track of moved files
    moved_files = {'used': [], 'unused': []}
    
    # Create separate dataframes for used and unused posters
    df_used = df[df['rank'] <= 50].copy()
    df_unused = df[df['rank'] > 50].copy()
    
    print("\nOrganizing posters into folders...")
    
    # Process each poster
    for _, row in df.iterrows():
        source_path = row['file_path']
        filename = row['filename']
        rank = row['rank']
        
        try:
            if rank <= 50:  # Top 50 go to used posters
                dest_path = os.path.join(used_dir, filename)
                moved_files['used'].append((filename, rank))
            else:  # Rest go to unused posters
                dest_path = os.path.join(unused_dir, filename)
                moved_files['unused'].append((filename, rank))
            
            # Copy file to new location
            shutil.copy2(source_path, dest_path)
            
        except Exception as e:
            print(f"Error moving file {filename}: {str(e)}")
    
    # Create timestamp for CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save folder-specific CSVs
    used_csv = os.path.join(used_dir, f"top_50_posters_{timestamp}.csv")
    unused_csv = os.path.join(unused_dir, f"remaining_posters_{timestamp}.csv")
    
    # Add methodology information to both CSVs
    methodology_info = pd.DataFrame({
        'Analysis Methodology': [
            'Poster Analysis Parameters:',
            '1. Color Features:',
            '   - RGB and HSV histograms (32 bins per channel)',
            '   - LAB color space moments (mean, std, skewness)',
            '   - Color variation in different color spaces',
            '',
            '2. Texture Features:',
            '   - Sobel gradient analysis (edges and details)',
            '   - GLCM (contrast, dissimilarity, homogeneity, energy, correlation)',
            '   - Local Binary Patterns (local texture patterns)',
            '',
            '3. Composition Features:',
            '   - Rule of thirds analysis',
            '   - Vertical and horizontal symmetry',
            '   - Edge density (visual complexity)',
            '',
            '4. Ranking Criteria:',
            '   - Color variation (30%): Diversity of colors in RGB and HSV spaces',
            '   - Visual appeal (20%): Brightness and saturation',
            '   - Texture quality (25%): Contrast and homogeneity',
            '   - Composition (15%): Edge density and visual complexity',
            '   - Balance (10%): Vertical and horizontal symmetry',
            '',
            '5. Additional Information:',
            '   - midjourney_prompt: Original prompt used to generate the image',
            f'   - Total analyzed: {len(df)} posters',
            '',
            'Data starts below:'
        ]
    })
    
    # Save CSVs with methodology information
    for csv_file, data in [(used_csv, df_used), (unused_csv, df_unused)]:
        with open(csv_file, 'w', encoding='utf-8') as f:
            methodology_info.to_csv(f, index=False)
            f.write('\n')
            data.to_csv(f, index=False)
    
    # Print summary
    print("\nPoster organization complete:")
    print(f"- Moved {len(moved_files['used'])} posters to 'used posters' folder")
    print(f"- Moved {len(moved_files['unused'])} posters to 'unused posters' folder")
    print(f"- Created CSV file for top 50 posters: {os.path.basename(used_csv)}")
    print(f"- Created CSV file for remaining posters: {os.path.basename(unused_csv)}")
    
    # Create a summary file
    summary_file = os.path.join(base_dir, "poster_organization_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Used Posters (Top 50):\n")
        f.write("-" * 50 + "\n")
        for filename, rank in sorted(moved_files['used'], key=lambda x: x[1]):
            f.write(f"Rank {rank}: {filename}\n")
        
        f.write("\nUnused Posters:\n")
        f.write("-" * 50 + "\n")
        for filename, rank in sorted(moved_files['unused'], key=lambda x: x[1]):
            f.write(f"Rank {rank}: {filename}\n")
    
    print(f"\nDetailed summary saved to: {summary_file}")
    
    return used_csv, unused_csv

def main():
    # Define the base directory
    base_dir = "/Users/tatiana/Downloads/Metaposters - Basketball Posters"
    image_dir = os.path.join(base_dir, "originals")
    
    # Analyze all posters
    results_df = analyze_all_posters(image_dir)
    
    if results_df is not None:
        total_images = len(results_df)
        print(f"\nAnalyzed and ranked all {total_images} posters.")
        print("\nTop 10 ranked posters:")
        for _, row in results_df.head(10).iterrows():
            print(f"Rank {row['rank']} of {total_images} (Score: {row['score']:.3f}):")
            print(f"Prompt: {row['midjourney_prompt']}\n")
        
        print(f"Bottom 10 ranked posters:")
        for _, row in results_df.tail(10).iterrows():
            print(f"Rank {row['rank']} of {total_images} (Score: {row['score']:.3f}):")
            print(f"Prompt: {row['midjourney_prompt']}\n")
        
        # Organize posters into appropriate folders
        used_csv, unused_csv = organize_posters_by_rank(results_df, base_dir)
        
        # Ask if user wants to display top posters
        display = input("\nWould you like to display the top 20 posters? (y/n): ")
        if display.lower() == 'y':
            top_20_paths = results_df.head(20)['file_path'].tolist()
            display_images(top_20_paths)

if __name__ == "__main__":
    main() 