# Snake-Assisted Food Segmentation

## Project Overview
This project implements a food segmentation system using active contour models (snakes) to identify and segment food items in images. The implementation is based on computer vision techniques for food image analysis.

## Key Features
- **Background Removal**: Uses histogram-based approach to identify and remove background from food images
- **Active Contour Models**: Implements the "snake" algorithm for food segmentation
- **Multiple Contour Initialization**: Automatically initializes multiple contours across the image for better segmentation

## Technical Details
The project uses several computer vision techniques:
1. **Background Removal**: Uses LAB color space and histogram binning to identify background pixels
2. **Contour Initialization**: Creates a grid of circular contours across the image
3. **Active Contour Model**: Uses the skimage implementation of active contours to evolve the initial contours to fit food boundaries

## Dependencies
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-image
- PIL (Python Imaging Library)
- Requests

## Installation

### Option 1: Local Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/snake-assisted-food-segmentation.git
   cd snake-assisted-food-segmentation
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Option 2: Google Colab
1. Open Google Colab: https://colab.research.google.com/
2. Upload the `main.py` file or copy its contents into a new notebook
3. Run the cells to execute the code

## Usage

### Running the Code
1. **Local Execution**:
   ```
   python main.py
   ```

   By default, the script uses a sample image. To use your own image, modify the image path in the script:
   ```python
   image_url = "path/to/your/image.jpg"  # Local path
   # or
   image_url = "https://example.com/image.jpg"  # URL
   ```

2. **Google Colab**:
   - To use your own images in Colab, uncomment and run the Google Drive mount code:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Then update the image path to point to your Drive:
     ```python
     image_url = "/content/drive/MyDrive/your_image.jpg"
     ```

### Adjusting Parameters
- Modify `num_initial_contours` to change the number of initial contours
- Adjust `background_bins` to fine-tune background removal
- Change visualization options in the `show_images` function

## Example Output
The script will display three images:
1. Original input image
2. Image with initial contours (red)
3. Background-removed image with final segmentation contours (blue)

## Evaluation Metrics

The project now includes evaluation metrics and visualizations to analyze the performance of the segmentation algorithm:

### Available Metrics

1. **Energy Convergence Visualization**: Plots how the energy function decreases during contour evolution
2. **Execution Time Analysis**: Measures and visualizes the time taken for each step of the segmentation process
3. **Parameter Sensitivity Analysis**: Analyzes how segmentation quality changes with different parameters


This will:
1. Download sample images if none are available
2. Perform detailed timing analysis of each segmentation step
3. Analyze segmentation performance with different parameter combinations
4. Visualize energy convergence during contour evolution


## References
This implementation is based on techniques from computer vision research in food image analysis and segmentation.
