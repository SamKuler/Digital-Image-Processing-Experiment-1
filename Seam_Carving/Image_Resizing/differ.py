import numpy as np
import sys
from ImageResizing import load_image, save_image, display_image

def compute_difference(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Compute the absolute difference between two images.
    
    :param image1: First input image as a NumPy array.
    :param image2: Second input image as a NumPy array.
    :return: Absolute difference image as a NumPy array.
    """
    return np.abs(image1.astype(np.int16) - image2.astype(np.int16))


def main():
    # Example usage
    i1_path = sys.path[0] + '/rider_output.png'
    i2_path = sys.path[0] + '/rider_resized.png'
    image1 = load_image(i1_path)
    image2 = load_image(i2_path)
    # Display the images
    display_image(image1)
    display_image(image2)
    # Compute the difference
    difference_image = compute_difference(image1, image2)
    # count non-zero pixels
    non_zero_count = np.count_nonzero(difference_image)
    print("Difference Image:")
    print(difference_image)
    print(f"Number of non-zero pixels: {non_zero_count}")
    # Display the difference image
    display_image(difference_image,True)
    # Save the difference image
    output_path = sys.path[0] + '/difference_image.png'
    save_image(difference_image, output_path)
if __name__ == "__main__":
    main()