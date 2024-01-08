# preprocess.py
import cv2
import os
import numpy as np

def add_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the image."""
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

def add_noise_to_folder(input_folder, output_folder, mean=0, sigma=25):
    """Add Gaussian noise to all images in the input folder and save to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".tif", ".bmp")):
                image_path = os.path.join(root, file)
                original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Add noise to the image
                noisy_image = add_noise(original_image, mean, sigma)

                # Save the noisy image to the output folder
                output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_noisy.tif")
                cv2.imwrite(output_path, noisy_image)

def rotate_image(image, angle):
    # Rotate the image by the specified angle
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def rotate_images_in_folder(folder_path, output_folder, angle):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".tif", ".bmp")):
                image_path = os.path.join(root, file)
                fingerprint_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Rotate the image by the specified angle
                rotated_image = rotate_image(fingerprint_image, angle)

                # Save rotated image to the output folder
                output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_rotated_{angle}.bmp")
                cv2.imwrite(output_path, rotated_image)

def preprocess_image(image_path, target_length):
    #print(f"Processing image: {image_path}")

    # Check if the file is an image
    if not image_path.lower().endswith((".tif", ".bmp")):
        print(f"Error: Unsupported image format for {image_path}")
        return None

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image does not exist at {image_path}")
        return None

    # Load the image
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None or image.size == 0:
        print(f"Error: Unable to load or empty image from {image_path}")
        return None

    # Check if the image has three channels (RGB)
    if image.shape[-1] != 3:
        print(f"Error: Unsupported image format (not RGB) for {image_path}")
        return None

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to be between 0 and 1
    normalized_image = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Resize the image
    resized_image = cv2.resize(normalized_image, (target_length, target_length))

    return resized_image

def extract_minutiae(image, target_length):
    # Convert the image to grayscale if it has three channels
    if image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Enhance the contrast using histogram equalization
    gray = cv2.equalizeHist(gray)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to obtain a binary image
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)

    # Find Harris corners (potential minutiae points)
    corners = cv2.cornerHarris(cleaned_image, blockSize=9, ksize=5, k=0.04)

    # Thresholding to extract minutiae points
    minutiae = np.argwhere(corners > 0.01 * corners.max())

    # Flatten and adjust the length
    minutiae_flat = minutiae.flatten()
    if len(minutiae_flat) < target_length * 2:
        minutiae_flat = np.pad(minutiae_flat, (0, target_length * 2 - len(minutiae_flat)))
    elif len(minutiae_flat) > target_length * 2:
        minutiae_flat = minutiae_flat[:target_length * 2]

    return minutiae_flat