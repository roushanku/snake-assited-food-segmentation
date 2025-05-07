import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.segmentation import active_contour
from collections import defaultdict
import requests
from io import BytesIO
from PIL import Image
import os

# Try to import Google Colab specific modules, but don't fail if not in Colab
try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

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


def draw_contours(image, contours, color, thickness=2):
    """
    Draw contours on an image
    Args:
        image: Background image
        contours: List of contours
        color: RGB tuple for contour color
        thickness: Contour line thickness
    Returns:
        Image with drawn contours
    """
    display = image.copy()
    for contour in contours:
        contour = contour.astype(int)
        cv2.polylines(display, [contour], True, color, thickness)
    return display



def calculate_energy(contour, gray_image):
    """
    Calculate the energy for a contour according to paper's Equations 1-3
    Args:
        contour: Current contour points (Nx2 array)
        gray_image: Grayscale image (h x w)
    Returns:
        Total energy (E_int + E_ext)
    """
    # Create mask for pixels inside contour
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [contour.astype(int)], 1)

    # Calculate mean intensities
    mu_in = np.mean(gray_image[mask == 1])
    mu_out = np.mean(gray_image[mask == 0])

    # Internal energy (Equation 1)
    diff_in = (gray_image - mu_in) ** 2
    E_int = np.sum(diff_in * mask)

    # External energy (Equation 2)
    diff_out = (gray_image - mu_out) ** 2
    E_ext = np.sum(diff_out * (1 - mask))

    # Total energy (Equation 3)
    return E_int + E_ext

def evolve_contour(contour, gray_image, max_iter=50, step_size=2.0):
    """
    Evolve contour to minimize energy
    Args:
        contour: Initial contour points
        gray_image: Input grayscale image
        max_iter: Maximum iterations
        step_size: Pixel movement step size
    Returns:
        evolved_contour: Final contour after evolution
        energy_history: List of energy values during evolution
    """
    current_contour = contour.copy()
    energy_history = []

    for _ in range(max_iter):
        current_energy = calculate_energy(current_contour, gray_image)
        energy_history.append(current_energy)

        # Generate candidate moves for each point
        candidates = []
        for i in range(len(current_contour)):
            for dx, dy in [(0, step_size), (0, -step_size),
                          (step_size, 0), (-step_size, 0)]:
                candidate = current_contour.copy()
                candidate[i] += [dx, dy]
                candidates.append(candidate)

        # Evaluate all candidates
        candidate_energies = [calculate_energy(c, gray_image) for c in candidates]
        best_idx = np.argmin(candidate_energies)

        # Stop if no improvement
        if candidate_energies[best_idx] >= current_energy:
            break

        current_contour = candidates[best_idx]

    return current_contour, energy_history

def segment_food(image, num_initial_contours=16, background_bins=32):
    """
    Complete food segmentation pipeline
    Args:
        image: Input RGB image
        num_initial_contours: Number of initial contours
        background_bins: Bins for background removal
    Returns:
        final_contours: List of final contours
        bg_removed: Image with background removed
    """
    # 1. Remove background
    bg_removed, bg_mask = remove_background(image, bins=background_bins)

    # 2. Convert to grayscale for segmentation
    gray = cv2.cvtColor(bg_removed, cv2.COLOR_RGB2GRAY).astype(float)
    gray = (gray - gray.min()) / (gray.max() - gray.min())  # Normalize

    # 3. Initialize contours
    initial_contours = initialize_contours(image.shape, num_initial_contours)

    # 4. Evolve contours
    final_contours = []
    for contour in initial_contours:
        # Skip contours entirely in background
        contour_points = contour.astype(int)
        if np.all(bg_mask[contour_points[:,1], contour_points[:,0]]):
            continue

        evolved_contour, _ = evolve_contour(contour, gray)
        final_contours.append(evolved_contour)

    return final_contours, bg_removed


def main():
    """Main function to run the segmentation"""
    # Check if sample_images directory exists, if not create it
    if not os.path.exists('sample_images'):
        os.makedirs('sample_images')
        print("Created sample_images directory. Please add some food images to this directory.")
        return

    # Get list of images in sample_images directory
    image_files = [f for f in os.listdir('sample_images') if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("No images found in sample_images directory. Please add some food images.")
        return

    # Use the first image
    image_path = os.path.join('sample_images', image_files[0])
    print(f"Using image: {image_path}")

    # Load image
    image = load_image(image_path)
    print("Original image size:", image.shape)

    # Run segmentation
    start_time = time.time()
    final_contours, bg_removed = segment_food(image, 4)
    total_time = time.time() - start_time
    print(f"Segmentation completed in {total_time:.2f} seconds")

    # Initialize contours for visualization
    initial_contours = initialize_contours(image.shape, 16)

    # Create visualizations
    initial_display = draw_contours(image, initial_contours, (255, 0, 0))  # Red
    final_display = draw_contours(bg_removed, final_contours, (0, 0, 255))  # Blue

    # Show results
    show_images([image, initial_display, final_display],
                ["Original Image", "Initial Contours", "Final Segmentation"])

    print("\nTo run evaluation metrics, use: python evaluation.py")

if __name__ == "__main__":
    main()