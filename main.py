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