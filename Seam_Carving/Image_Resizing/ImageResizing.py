import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from tqdm import trange
import numba as nb
from typing import Optional
import re
import time
# import cv2
# from scipy.ndimage import sobel

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path and convert it to a NumPy array.
    
    :param image_path: Path to the image file.
    :return: Image as a NumPy array.
    """
    image = Image.open(image_path)
    return np.array(image)

def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save a NumPy array as an image file.
    
    :param image: Image as a NumPy array.
    :param output_path: Path to save the image file.
    """
    img = Image.fromarray(image.astype('uint8'))
    img.save(output_path)


def display_image(image: np.ndarray,grayscale : bool = False) -> None:
    """
    Display an image using matplotlib.
    
    :param image: Image as a NumPy array.
    :param grayscale: If True, display the image in grayscale.
    """
    plt.imshow(image, cmap='gray' if grayscale else None)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def rgb2gray(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.
    
    :param rgb_image: RGB image as a NumPy array.
    :return: Grayscale image as a NumPy array.
    """
    if rgb_image.ndim == 2:
        return rgb_image.astype(np.float32)
    if rgb_image.shape[2] == 4:  # If the image has an alpha channel
        rgb_image = rgb_image[..., :3]
    if rgb_image.shape[2] != 3:
        raise ValueError("Input image must be RGB or RGBA format.")
    coeffs = np.array([0.2989, 0.5870, 0.1140],dtype=np.float32)  # Coefficients for RGB to grayscale conversion
    # Convert to grayscale using the coefficients
    return (rgb_image @ coeffs).astype(np.float32)
    # return np.average(rgb_image, axis=2).astype(np.float32)  # Using average for simplicity

def transpose_image(image: np.ndarray) -> np.ndarray:
    """
    Transpose an image (swap rows and columns).
    
    :param image: Input image as a NumPy array.
    :return: Transposed image as a NumPy array.
    """
    return np.transpose(image, (1, 0, 2)) if image.ndim == 3 else np.transpose(image)

def cal_energy_map(image: np.ndarray, grayscale: bool = False) -> np.ndarray:
    """
    Compute the energy map of an image using the Sobel operator.
    
    :param image: Input image as a NumPy array.
    :param grayscale: If True, compute energy on grayscale image.
    :return: Energy map as a NumPy array.
    """
    if not grayscale:
        image = rgb2gray(image)
    
    energy_x = np.zeros_like(image)
    energy_x[:, 1:-1] = np.abs(image[:, 2:] - image[:, :-2])
    energy_x[:, 0] = np.abs(image[:, 1] - image[:, 0])
    energy_x[:, -1] = np.abs(image[:, -1] - image[:, -2])
    
    energy_y = np.zeros_like(image)
    energy_y[1:-1, :] = np.abs(image[2:, :] - image[:-2, :])
    energy_y[0, :] = np.abs(image[1, :] - image[0, :])
    energy_y[-1, :] = np.abs(image[-1, :] - image[-2, :])

    # energy_x2 = np.abs(cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3))
    # energy_y2 = np.abs(cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3))
    # assert energy_x.all()  == energy_x2.all(), "Energy map calculation using Sobel operator is incorrect."
    # assert energy_y.all() == energy_y2.all(), "Energy map calculation using Sobel operator is incorrect."
    
    # energy_x = np.abs(sobel(image, axis=1))
    # energy_y = np.abs(sobel(image, axis=0))

    return energy_x + energy_y

def mark_seam(image:np.ndarray, seam: np.ndarray, direction: str = 'vertical',color = (255,0,0)) -> np.ndarray:
    """
    Mark a seam on the image.
    
    :param image: Input image as a NumPy array.
    :param seam: Indices of the pixels in the seam.
    :param direction: Direction of the seam ('vertical' or 'horizontal').
    :param color: Color to mark the seam.
    :return: Image with the seam marked.
    """
    if direction not in ['vertical', 'horizontal']:
        raise ValueError("Direction must be 'vertical' or 'horizontal'.")
    marked_image = image.copy()
    rows, cols = marked_image.shape[:2]
    if color is None:
        color = (255, 0, 0)
    if image.ndim == 2:
        for i in range(rows):
            if direction == 'vertical':
                marked_image[i, seam[i]] = color[0]
            elif direction == 'horizontal':
                marked_image[seam[i], i] = color[0]
    else:
        for i in range(rows):
            if direction == 'vertical':
                marked_image[i, seam[i], :3] = color
            elif direction == 'horizontal':
                marked_image[seam[i], i, :3] = color
    return marked_image

def mark_seams(image: np.ndarray, seams_mask: np.ndarray, color: tuple = (255, 0, 0)) -> np.ndarray:
    """
    Mark multiple seams on the image.
    
    :param image: Input image as a NumPy array.
    :param seams_mask: Mask of seams as a NumPy array.
    :param color: Color to mark the seams.
    :return: Image with the seams marked.
    """
    marked_image = image.copy()
    rows, cols = marked_image.shape[:2]
    if image.ndim == 2:
        for i in range(rows):
            for j in range(cols):
                if seams_mask[i, j]:
                    marked_image[i, j] = color[0]
    else:
        for i in range(rows):
            for j in range(cols):
                if seams_mask[i, j]:
                    marked_image[i, j, :3] = color
    return marked_image

def seam2seams(seam: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert a seam to a seams mask.
    
    :param seam: Seam (vertical) as a NumPy array.
    :param shape: Shape of the image (rows, cols).
    :return: Seams mask as a NumPy array.
    """
    # Select the corresponding columns in the identity matrix for each row
    return np.eye(shape[1], dtype=bool)[seam]

def remove_by_seams(image: np.ndarray, seam_mask:np.ndarray,seam_num:Optional[int]=None) -> np.ndarray:
    """
    Remove seams from the image.
    
    :param image: Input image as a NumPy array.
    :param seam_mask: Mask of seams as a NumPy array.
    :param seam_num: Number of seams to remove. If None, it will be calculated from the seam mask.
    :return: Image with the seams removed.
    """
    # removed= np.zeros((image.shape[0], image.shape[1] - 1), dtype=image.dtype)
    # if image.ndim == 2:
    #     for i in range(image.shape[0]):
    #         removed[i] = np.delete(image[i], seam_mask[i])
    # else:
    #     for i in range(image.shape[0]):
    #         removed[i] = np.delete(image[i], seam_mask[i], axis=1)
    # return removed
    # Faster implementation using NumPy
    if seam_num is None:
        seam_num = np.count_nonzero(seam_mask[0])
    if image.ndim == 2:
        removed = image[~seam_mask].reshape(image.shape[0], image.shape[1] - seam_num)
    else:
        seam_mask = np.broadcast_to(seam_mask[:, :, np.newaxis], image.shape)
        removed = image[~seam_mask].reshape(image.shape[0], image.shape[1] - seam_num, image.shape[2])
    return removed


@nb.njit(nb.int32[:](nb.float32[:,:]), cache=True)
def find_single_seam(energy_map: np.ndarray) -> np.ndarray:
    """
    Find the seam with the lowest energy in the energy map.
    
    :param energy_map: Energy map as a NumPy array.
    :return: Indices of the pixels in the seam.
    """
    rows, cols = energy_map.shape
    
    cumulated_energy=energy_map.copy()
    path=np.zeros((rows, cols), dtype=np.int32)
    for i in range(1, rows):
        for j in range(cols):
            left = cumulated_energy[i-1, j-1] if j > 0 else np.inf
            up = cumulated_energy[i-1, j]
            right = cumulated_energy[i-1, j+1] if j < cols - 1 else np.inf
            
            min_energy = min(left, up, right)
            cumulated_energy[i, j] += min_energy
            if min_energy == left:
                path[i, j] = j - 1
            elif min_energy == right:
                path[i, j] = j + 1
            else:
                path[i, j] = j
        
    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = np.argmin(cumulated_energy[-1])
    for i in range(rows - 2, -1, -1):
        seam[i] = path[i + 1, seam[i + 1]]
    return seam

def find_seams(grayscale_image: np.ndarray, num_seams: int,energy_mask: Optional[np.ndarray]=None) -> np.ndarray:
    """
    Find multiple seams with the lowest energy in the energy map.
    
    :param grayscale_image: Grayscale image as a NumPy array.
    :param energy_mask: Energy mask as a NumPy array.
    :param num_seams: Number of seams to find.
    :return: seams mask as a NumPy array of shape (rows, cols).
    """
    rows,cols = grayscale_image.shape
    seams_mask = np.zeros((rows, cols), dtype=bool)# Create a mask to mark the seams
    rows_arange = np.arange(rows)# Create an array of row indices to avoid repeated calls to np.arange
    index_map = np.broadcast_to(np.arange(cols), (rows, cols))# Save current index for correctly remapping seams after removing them
    energy_map = cal_energy_map(grayscale_image)
    if energy_mask is not None:
        energy_map += energy_mask
    
    if num_seams==1:
        single_seam = find_single_seam(energy_map)
        seams_mask[rows_arange, index_map[rows_arange, single_seam]] = True
        return seams_mask

    for _ in trange(num_seams):
        # Find the first seam
        single_seam = find_single_seam(energy_map)
        seams_mask[rows_arange, index_map[rows_arange, single_seam]] = True

        # Update related information
        single_seam_mask = seam2seams(single_seam, energy_map.shape)
        grayscale_image = remove_by_seams(grayscale_image, single_seam_mask,1)
        index_map = remove_by_seams(index_map, single_seam_mask,1)
        if energy_mask is not None:
            energy_mask = remove_by_seams(energy_mask, single_seam_mask,1)

        # Update the energy map based on the seam
        #1 Find boundaries of the seam
        cur_rows,cur_cols = energy_map.shape
        left = max(0, np.min(single_seam) - 1)
        right = min(cur_cols, np.max(single_seam) + 1)
        # Extend the boundaries again to ensure pixels in the boundaries are enough to update the energy map
        left_extd = 1 if left > 0 else 0
        right_extd = 1 if right < cur_cols - 1 else 0
        #2 Split the energy map and calculate the energy map of the middle block      
        mid_block = grayscale_image[:, left - left_extd : right + right_extd]
        _, mid_block_col = mid_block.shape
        mid_energy = cal_energy_map(mid_block)[:, left_extd : mid_block_col - right_extd]
        #3 Update the energy map
        if energy_mask is not None:
            mid_energy += energy_mask[:, left:right]
        energy_map = np.hstack((energy_map[:, :left], mid_energy, energy_map[:, right + 1 :]))

    return seams_mask

@nb.njit(cache=True)
def insert_by_seams(image: np.ndarray, seam_mask:np.ndarray,seam_num:Optional[int]=None) -> np.ndarray:
    """
    Insert seams into the image.
    
    :param image: Input image as a NumPy array.
    :param seam_mask: Mask of seams as a NumPy array.
    :param seam_num: Number of seams to insert. If None, it will be calculated from the seam mask.
    :return: Image with the seams inserted.
    """
    if seam_num is None:
        added = np.count_nonzero(seam_mask[0])
    else:
        added = seam_num
    if image.ndim == 2:
        inserted = np.empty((image.shape[0], image.shape[1] + added), dtype=image.dtype)
    else:
        inserted = np.empty((image.shape[0], image.shape[1] + added, image.shape[2]), dtype=image.dtype)
    rows, cols = image.shape[:2]
    for i in range(rows):
        inserted_num = 0
        for j in range(cols):
            if seam_mask[i, j]:
                # Linear interpolation
                left = image[i, max(0, j - 1)]
                right = image[i, j]
                inserted[i, j + inserted_num] = (left + right) / 2
                inserted_num += 1
            inserted[i, j + inserted_num] = image[i, j]

    return inserted

def reduce_image_width(image: np.ndarray, dst_width: int, energy_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reduce the width of an image using seam carving.
    
    :param image: Input image as a NumPy array.
    :param dst_width: Desired width of the output image.
    :param energy_mask: Energy mask as a NumPy array.
    :return: Resized image as a NumPy array.
    """
    assert image.ndim in [2, 3], "Image must be either grayscale or RGB/RGBA."
    assert dst_width < image.shape[1], "Destination width must be less than the original width."
    if image.ndim == 2:
        grayscale_image = image
        dst_shape = (image.shape[0], dst_width)
    else:
        grayscale_image = rgb2gray(image)
        dst_shape = (image.shape[0], dst_width, image.shape[2])

    remaining_mask = ~find_seams(grayscale_image, image.shape[1] - dst_width,energy_mask)
    
    resized_image = image[remaining_mask].reshape(dst_shape)
    if energy_mask is not None:
        energy_mask = energy_mask[remaining_mask].reshape(dst_shape[:2])
    
    return resized_image, energy_mask

def reduce_image_height(image: np.ndarray, dst_height: int, energy_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reduce the height of an image using seam carving.
    
    :param image: Input image as a NumPy array.
    :param dst_height: Desired height of the output image.
    :param energy_mask: Energy mask as a NumPy array.
    :return: Resized image as a NumPy array.
    """
    assert image.ndim in [2, 3], "Image must be either grayscale or RGB/RGBA."
    assert dst_height < image.shape[0], "Destination height must be less than the original height."
    image = transpose_image(image)
    if energy_mask is not None:
        energy_mask = transpose_image(energy_mask)
    resized_image, energy_mask = reduce_image_width(image, dst_height, energy_mask)
    resized_image = transpose_image(resized_image)
    if energy_mask is not None:
        energy_mask = transpose_image(energy_mask)
    return resized_image, energy_mask

def expand_image_width(image: np.ndarray, dst_width: int, energy_mask: Optional[np.ndarray] = None,gap:float=0.5) -> np.ndarray:
    """
    Expand an image width by seams

    :param image: Input image as a NumPy array.
    :param dst_width: Desired width of the output image.
    :param energy_mask: Energy mask as a NumPy array.
    :param gap: Each enlargement loop will insert at most gap*width seams
    :return: Expanded image as a NumPy array
    """
    assert image.ndim in [2, 3], "Image must be either grayscale or RGB/RGBA."
    assert dst_width > image.shape[1], "Destination width must be greater than the original width."
    
    expanded = image.copy()

    while expanded.shape[1] < dst_width:
        num_seams = min(max(1, round(expanded.shape[1]* gap)), dst_width - expanded.shape[1])#Ensure the num_seams is not too large
        grayscale_image = rgb2gray(expanded) if expanded.ndim == 3 else expanded
        seams_mask = find_seams(grayscale_image, num_seams, energy_mask)
        expanded = insert_by_seams(expanded, seams_mask,num_seams)
        if energy_mask is not None:
            energy_mask = insert_by_seams(energy_mask, seams_mask,num_seams)
    
    return expanded, energy_mask


def expand_image_height(image: np.ndarray, dst_height: int, energy_mask: Optional[np.ndarray] = None,gap:float=0.5) -> np.ndarray:
    """
    Expand an image height by seams

    :param image: Input image as a NumPy array.
    :param dst_height: Desired height of the output image.
    :param energy_mask: Energy mask as a NumPy array.
    :param gap: Each enlargement loop will insert at most gap*height seams
    :return: Expanded image as a NumPy array
    """
    assert image.ndim in [2, 3], "Image must be either grayscale or RGB/RGBA."
    assert dst_height > image.shape[0], "Destination height must be greater than the original height."
    
    expanded = image.copy()
    expanded = transpose_image(expanded)
    if energy_mask is not None:
        energy_mask = transpose_image(energy_mask)
    expanded,energy_mask = expand_image_width(expanded, dst_height, energy_mask, gap)
    expanded = transpose_image(expanded)
    if energy_mask is not None:
        energy_mask = transpose_image(energy_mask)
    return expanded, energy_mask

def greedy_resize(image: np.ndarray, dst_shape: tuple, energy_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Resize an image using greedy seam direction selection (minimum energy) at each step.
    Not available for expanding image.
    
    :param image: Input image as a NumPy array
    :param dst_shape: Desired shape (rows, cols)
    :param energy_mask: Energy mask as a NumPy array
    :return: Resized image
    """
    assert image.ndim in [2, 3], "Image must be either grayscale or RGB/RGBA."
    assert len(dst_shape) == 2, "Destination shape must be a tuple of (rows, cols)."
    dst_shape = (round(dst_shape[0]), round(dst_shape[1]))
    assert dst_shape[0] <= image.shape[0], "Destination height must be less than the original height."
    assert dst_shape[1] <= image.shape[1], "Destination width must be less than the original width."
    
    resized_image = image.copy()
    cur_mask = energy_mask.copy() if energy_mask is not None else None
    pbar = trange(abs(resized_image.shape[0] - dst_shape[0]) + abs(resized_image.shape[1] - dst_shape[1]), desc="Resizing")
    pbar.refresh()
    while (resized_image.shape[0] != dst_shape[0]) or (resized_image.shape[1] != dst_shape[1]):
        h_energy = float('inf')#Horizontal Seam Energy
        v_energy = float('inf')#Vertical Seam Energy
        
        # Consider vertical seam (width change) if needed
        if resized_image.shape[1] != dst_shape[1]:
            gray = rgb2gray(resized_image) if resized_image.ndim == 3 else resized_image
            energy_map_v = cal_energy_map(gray)
            if cur_mask is not None:
                energy_map_v += cur_mask
            v_seam = find_single_seam(energy_map_v)
            
            # Get minimum cumulative energy for this seam
            v_energy = energy_map_v[-1, v_seam[-1]]
        
        # Consider horizontal seam (height change) if needed
        if resized_image.shape[0] != dst_shape[0]:
            transposed = transpose_image(resized_image)
            trans_mask = transpose_image(cur_mask) if cur_mask is not None else None
            gray_t = rgb2gray(transposed) if transposed.ndim == 3 else transposed
            energy_map_h = cal_energy_map(gray_t)
            if trans_mask is not None:
                energy_map_h += trans_mask
            h_seam = find_single_seam(energy_map_h)
            
            # Get minimum cumulative energy for this seam
            h_energy = energy_map_h[-1, h_seam[-1]]
        
        # Choose the direction with the minimum energy seam
        if (h_energy < v_energy and resized_image.shape[0] != dst_shape[0]) or resized_image.shape[1] == dst_shape[1]:
            # Remove horizontal seam (change height)
            resized_image = transpose_image(resized_image)
            resized_image = remove_by_seams(resized_image, seam2seams(h_seam, resized_image.shape), 1)
            resized_image = transpose_image(resized_image)
            if cur_mask is not None:
                cur_mask = transpose_image(cur_mask)
                cur_mask = remove_by_seams(cur_mask, seam2seams(h_seam, resized_image.shape), 1)
                cur_mask = transpose_image(cur_mask)
        elif resized_image.shape[1] != dst_shape[1]:
            # Remove vertical seam (change width)
            resized_image = remove_by_seams(resized_image, seam2seams(v_seam, resized_image.shape), 1)
            if cur_mask is not None:
                cur_mask = remove_by_seams(cur_mask, seam2seams(v_seam, resized_image.shape), 1)
            
        pbar.update(1)
        pbar.refresh()

    return resized_image


def image_resizing(image:np.ndarray,dst_shape:tuple,energy_mask:Optional[np.ndarray] = None,priority:str='cols') -> np.ndarray:
    """
    Resize an image using seam carving.
    
    :param image: Input image as a NumPy array.
    :param dst_shape: Desired shape of the output image (rows, cols).
    :param energy_mask: Energy mask as a NumPy array.
    :param priority: Priority for resizing ('rows', 'cols', 'alternate', or 'greedy').
    :return: Resized image as a NumPy array.
    """
    assert image.ndim in [2, 3], "Image must be either grayscale or RGB/RGBA."
    assert len(dst_shape) == 2, "Destination shape must be a tuple of (rows, cols)."
    assert priority in ['cols','rows','alternate', 'greedy'], "Priority must be 'rows', 'cols', 'alternate', or 'greedy'."
    resized_image = image.copy()
    
    dst_shape= (round(dst_shape[0]), round(dst_shape[1]))

    if priority=='cols':
        if dst_shape[1] < resized_image.shape[1]:
            resized_image, energy_mask = reduce_image_width(resized_image, dst_shape[1], energy_mask)
        elif dst_shape[1] > resized_image.shape[1]:
            resized_image, energy_mask = expand_image_width(resized_image, dst_shape[1], energy_mask)

        if dst_shape[0] < resized_image.shape[0]:
            resized_image, energy_mask = reduce_image_height(resized_image, dst_shape[0], energy_mask)
        elif dst_shape[0] > resized_image.shape[0]:
            resized_image, energy_mask = expand_image_height(resized_image, dst_shape[0], energy_mask)

    elif priority=='rows':
        if dst_shape[0] < resized_image.shape[0]:
            resized_image, energy_mask = reduce_image_height(resized_image, dst_shape[0], energy_mask)
        elif dst_shape[0] > resized_image.shape[0]:
            resized_image, energy_mask = expand_image_height(resized_image, dst_shape[0], energy_mask)
        if dst_shape[1] < resized_image.shape[1]:
            resized_image, energy_mask = reduce_image_width(resized_image, dst_shape[1], energy_mask)
        elif dst_shape[1] > resized_image.shape[1]:
            resized_image, energy_mask = expand_image_width(resized_image, dst_shape[1], energy_mask)

    elif priority=='alternate':
        delta = abs(resized_image.shape[0] - dst_shape[0]) + abs(resized_image.shape[1] - dst_shape[1])
        for _ in trange(delta):
            if dst_shape[1] < resized_image.shape[1]:
                resized_image, energy_mask = reduce_image_width(resized_image, resized_image.shape[1]-1, energy_mask)
            elif dst_shape[1] > resized_image.shape[1]:
                resized_image, energy_mask = expand_image_width(resized_image, resized_image.shape[1]+1, energy_mask)
            if dst_shape[0] < resized_image.shape[0]:
                resized_image, energy_mask = reduce_image_height(resized_image, resized_image.shape[0]-1, energy_mask)
            elif dst_shape[0] > resized_image.shape[0]:
                resized_image, energy_mask = expand_image_height(resized_image, resized_image.shape[0]+1, energy_mask)
    elif priority=='greedy':
        resized_image = greedy_resize(resized_image, dst_shape, energy_mask)
    else:
        raise ValueError("Invalid priority value. Use 'rows', 'cols', 'alternate', or 'greedy'.")

    return resized_image

def object_remove(image:np.ndarray,keep_mask:np.ndarray,delete_mask:np.ndarray,energy_mask:Optional[np.ndarray]=None,priority:str='cols') -> np.ndarray:
    # Implemention in dip
    pass

INFO_IMAGE_RESIZING="""
============================================================ 
=           Image Resizing Tool By Seam Carving            =
============================================================
"""
INFO_OBJECT_REMOVE="""
============================================================
=           Object Removal Tool By Seam Carving            =
============================================================
"""
def image_resizing_ui():
    # UI
    print(INFO_IMAGE_RESIZING)
    image_path = input("Please input the image path:").strip(' \'\"\n\r')
    while not image_path:
        print("Image path cannot be empty.")
        image_path = input("Please input the image path:").strip(' \'\"\n\r')
    image = load_image(image_path)
    print(f"Your image is with shape (rows, cols, channels): {image.shape}")


    dst_shape = input("Please input the destination shape (rows, cols):").strip()
    while not dst_shape:
        print("Destination shape cannot be empty.")
        dst_shape = input("Please input the destination shape (rows, cols):").strip()
    dst_shape = re.findall(r'\d+', dst_shape)
    dst_shape = tuple(map(int, dst_shape))

    energy_mask = input("Please input the energy mask path (or leave blank): ").strip(' \'\"\n\r')
    if energy_mask:
        energy_mask = load_image(energy_mask)
    else:
        energy_mask = None
    priority = input("Please input the resizing priority ('rows', 'cols', 'alternate', 'greedy', or leave blank for 'cols' priority:").strip()
    while priority not in ['rows', 'cols', 'alternate', 'greedy', '']:
        priority = input("Invalid priority. Please input again ('rows', 'cols', 'alternate', 'greedy', or leave blank):").strip()
    if priority == '':
        priority = 'cols'


    start_time = time.perf_counter()
    resized_image = image_resizing(image, dst_shape, energy_mask, priority)
    end_time = time.perf_counter()

    print(f"Processed with {priority} priority in {end_time - start_time:.6f} seconds.")

    output_path = input("Please input the output path, or leave blank to use the default path (./*image*_resized.png/jpg/etc.): ").strip(' \'\"\n\r')
    if not output_path:
        split = image_path.split('/')
        output_path = "/".join(split[:-1]) + "/" + split[-1].split('.')[0] + "_resized." + split[-1].split('.')[-1]
    save_image(resized_image, output_path)
    print("Resizing completed. The resized image is saved at:", output_path)
    display_image(resized_image, grayscale=(resized_image.ndim == 2))

if __name__ == "__main__":
    image_resizing_ui()
    