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

## Usage
The code is designed to work in a Google Colab environment, but can be adapted for local use. It can process images from local paths or URLs.

## References
This implementation is based on techniques from computer vision research in food image analysis and segmentation.
