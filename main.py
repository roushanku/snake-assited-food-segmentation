import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from skimage.segmentation import active_contour
from collections import defaultdict
import requests
from io import BytesIO
from PIL import Image

# Mount Google Drive if using your own images
# from google.colab import drive
# drive.mount('/content/drive')
def load_image(path_or_url, downsample_factor=4):
    """Load image from path or URL and preprocess"""
    if path_or_url.startswith('http'):
        response = requests.get(path_or_url)
        img = Image.open(BytesIO(response.content))
        img = np.array(img)
    else:
        img = cv2.imread(path_or_url)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # # Downsample as mentioned in paper
    # if downsample_factor > 1:
    #     h, w = img.shape[:2]
    #     img = cv2.resize(img, (w//downsample_factor, h//downsample_factor))

    return img

def show_images(images, titles=None, figsize=(15, 5)):
    """Display multiple images"""
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img)
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

image = load_image("/content/8007a59.jpg")
show_images([image])


def remove_background(image, bins=32, background_color=(255, 255, 255), threshold_factor=1.5):
    """
    Remove background using the paper's histogram approach with improvements
    Args:
        image: Input RGB image
        bins: Number of bins per channel (paper uses 32)
        background_color: Color to replace background with
        threshold_factor: Factor to determine background frequency threshold
    Returns:
        bg_removed: Image with background removed
        background_mask: Binary mask of background regions
    """
    # Convert to LAB color space for better color differentiation
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    pixels = lab_image.reshape(-1, 3)

    # Calculate bin indices for each pixel
    bin_size = 256 // bins
    bin_indices = (pixels // bin_size).astype(int)

    # Clip to ensure indices are within range
    bin_indices = np.clip(bin_indices, 0, bins-1)

    # Find most frequent bin combination
    bin_counts = defaultdict(int)
    for l, a, b in bin_indices:
        bin_counts[(l, a, b)] += 1

    # Sort bins by frequency
    sorted_bins = sorted(bin_counts.items(), key=lambda x: x[1], reverse=True)

    # Get the most frequent bin (likely background)
    background_bin = sorted_bins[0][0]
    background_count = sorted_bins[0][1]

    # Calculate threshold for similar bins (consider them as background too)
    threshold = background_count / threshold_factor

    # Find bins that are similar to the background bin (likely also background)
    background_bins = [background_bin]
    for bin_key, count in sorted_bins[1:5]:  # Check the next few most frequent bins
        if count > threshold:
            # Check if the bin is similar to the background bin
            l_diff = abs(bin_key[0] - background_bin[0])
            a_diff = abs(bin_key[1] - background_bin[1])
            b_diff = abs(bin_key[2] - background_bin[2])

            # If the bin is similar in color to the background bin, add it
            if l_diff <= 1 and a_diff <= 1 and b_diff <= 1:
                background_bins.append(bin_key)

    # Convert back to original shape
    bin_indices_reshaped = bin_indices.reshape(image.shape[:2] + (3,))

    # Create background mask - pixels in any of the background bins
    background_mask = np.zeros(image.shape[:2], dtype=bool)
    for bg_bin in background_bins:
        mask_l = bin_indices_reshaped[:, :, 0] == bg_bin[0]
        mask_a = bin_indices_reshaped[:, :, 1] == bg_bin[1]
        mask_b = bin_indices_reshaped[:, :, 2] == bg_bin[2]
        bin_mask = mask_l & mask_a & mask_b
        background_mask = background_mask | bin_mask

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    background_mask = background_mask.astype(np.uint8)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)

    # Fill holes in the background mask
    # Invert the mask to find holes
    inverted_mask = 1 - background_mask
    # Label connected components
    num_labels, labels = cv2.connectedComponents(inverted_mask)

    # If a component doesn't touch the border, it's a hole
    for label in range(1, num_labels):
        component = (labels == label)
        # Check if component touches the border
        if not (np.any(component[0, :]) or np.any(component[-1, :]) or
                np.any(component[:, 0]) or np.any(component[:, -1])):
            # It's a hole, fill it
            background_mask[component] = 1

    # Convert back to boolean
    background_mask = background_mask.astype(bool)

    # Replace background
    bg_removed = image.copy()
    bg_removed[background_mask] = background_color

    return bg_removed, background_mask


def initialize_contours(image_shape, num_contours=8):
    """
    Initialize multiple circular contours across the image
    Args:
        image_shape: Shape of the input image (h, w, c)
        num_contours: Number of contours to create (paper used 9-289)
    Returns:
        List of initialized contours (each contour is Nx2 array)
    """
    h, w = image_shape[:2]
    contours = []

    # Calculate grid size (nearest perfect square <= num_contours)
    grid_size = int(np.sqrt(num_contours))
    actual_num = grid_size ** 2

    # Spacing between contours
    x_spacing = w // (grid_size + 1)
    y_spacing = h // (grid_size + 1)

    # Radius of each circular contour
    radius = min(x_spacing, y_spacing) // 3

    # Create circular contours at grid points
    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):
            center_x = j * x_spacing
            center_y = i * y_spacing

            # Generate points on circle
            theta = np.linspace(0, 2*np.pi, 50)
            x = center_x + radius * np.cos(theta)
            y = center_y + radius * np.sin(theta)

            contours.append(np.column_stack((x, y)))

    return contours