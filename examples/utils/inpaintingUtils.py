import cv2
import numpy as np
import heapq
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import distance_transform_edt

# Constants
KNOWN = 0    # Pixels outside the mask
BAND = 1     # Narrow band (pixels near the mask)
INSIDE = 2   # Pixels inside the mask (to be inpainted)

def fast_marching_inpaint(image, mask, inpaint_radius=5):
    """
    Python implementation of OpenCV's Fast Marching Inpainting.
    
    Parameters:
        image (np.ndarray): 16-bit or 8-bit grayscale input image.
        mask (np.ndarray): Binary mask (255 where inpainting is needed).
        inpaint_radius (int): The radius of influence for the inpainting.
        
    Returns:
        np.ndarray: Inpainted image.
    """
    # Convert input to float32 for precision
    image = image.astype(np.float32)
    
    # Step 1: Compute distance transform (to find narrow band)
    distance, indices = distance_transform_edt(mask == 0, return_indices=True)

    # Step 2: Initialize Fast Marching Priority Queue
    priority_queue = []
    status = np.full_like(mask, INSIDE, dtype=np.uint8)  # Mark all as "inside"
    
    # Mark known pixels
    status[mask == 0] = KNOWN  

    # Find the narrow band pixels (BAND) and push them into the priority queue
    band_pixels = np.argwhere((mask > 0) & (distance < inpaint_radius))
    for y, x in band_pixels:
        heapq.heappush(priority_queue, (distance[y, x], y, x))
        status[y, x] = BAND  

    # Step 3: Process the priority queue (Fast Marching)
    while priority_queue:
        _, y, x = heapq.heappop(priority_queue)

        # Skip if already processed
        if status[y, x] == KNOWN:
            continue
        
        # Find 4-connected neighbors
        neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
        valid_neighbors = [(ny, nx) for ny, nx in neighbors if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]]

        # Compute weighted average of known pixels
        known_values = [image[ny, nx] for ny, nx in valid_neighbors if status[ny, nx] == KNOWN]
        if known_values:
            image[y, x] = np.mean(known_values)

        # Mark as known
        status[y, x] = KNOWN

        # Push neighbors into the queue
        for ny, nx in valid_neighbors:
            if status[ny, nx] == INSIDE:
                heapq.heappush(priority_queue, (distance[ny, nx], ny, nx))
                status[ny, nx] = BAND  

    return image.astype(np.uint16)


def directional_inpaint(ir_image, binary_mask, contours):
    """
    Performs directional inpainting on a 16-bit IR image in parallel:
    - Left half of each blob is filled using only left border pixels.
    - Right half of each blob is filled using only right border pixels.

    Parameters:
        ir_image (numpy.ndarray): 16-bit IR image.
        binary_mask (numpy.ndarray): 8-bit binary mask of blobs to be inpainted.
        contours (list): List of contours representing blobs.

    Returns:
        numpy.ndarray: Inpainted 16-bit image with directional filling.
    """
    # Copy original image to store final inpainted results
    final_inpainted = ir_image.copy()

    # **Parallel Execution Using ThreadPoolExecutor**
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda cnt: inpaint_single_blob(ir_image, cnt, binary_mask), contours))
    
    # **Merge Results inpaint_single_blob1**
    for left_inpainted, right_inpainted, left_mask, right_mask in results:
        final_inpainted[left_mask > 0] = left_inpainted[left_mask > 0]
        final_inpainted[right_mask > 0] = right_inpainted[right_mask > 0]
    
  
    return final_inpainted.astype(np.uint16)

def directional_inpaint(ir_image, binary_mask, contours):
    """
    Performs directional inpainting on a 16-bit IR image in parallel:
    - Left half of each blob is filled using only left border pixels.
    - Right half of each blob is filled using only right border pixels.

    Parameters:
        ir_image (numpy.ndarray): 16-bit IR image.
        binary_mask (numpy.ndarray): 8-bit binary mask of blobs to be inpainted.
        contours (list): List of contours representing blobs.

    Returns:
        numpy.ndarray: Inpainted 16-bit image with directional filling.
    """
    # Copy original image to store final inpainted results
    final_inpainted = ir_image.copy()

    # **Parallel Execution Using ThreadPoolExecutor**
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda cnt: inpaint_single_blob(ir_image, cnt, binary_mask), contours))
    
    # **Merge Results inpaint_single_blob1**
    for left_inpainted, right_inpainted, left_mask, right_mask in results:
        final_inpainted[left_mask > 0] = left_inpainted[left_mask > 0]
        final_inpainted[right_mask > 0] = right_inpainted[right_mask > 0]
    
  
    return final_inpainted.astype(np.uint16)


def inpaint_single_blob(ir_image, cnt, binary_mask):
    """
    Inpaints a single blob by splitting it into left and right halves.
    This function is meant to run in parallel.
    """
    # Create separate masks
    left_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    right_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # Get bounding box of the blob
    x, y, w, h = cv2.boundingRect(cnt)
    center_x = x + w // 2  # Compute vertical center of the blob

    # Draw the contour onto the masks
    cv2.drawContours(left_mask, [cnt], -1, 255, thickness=-1)
    cv2.drawContours(right_mask, [cnt], -1, 255, thickness=-1)

    
    # Keep only left and right halves separately
    left_mask[:, center_x:] = 0   
    right_mask[:, :center_x] = 0  

    # Create "clean" copies of the image
    ir_left = ir_image.copy()
    ir_right = ir_image.copy()

     # **Step 2: Pre-Fill Missing Pixels Before Inpainting**
    ir_left[right_mask > 0] = 0 
    ir_right[left_mask > 0] = 0 

    # Perform inpainting
    left_inpainted = cv2.inpaint(ir_left, left_mask, inpaintRadius=5,flags=cv2.INPAINT_TELEA)
    right_inpainted = cv2.inpaint(ir_right, right_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return left_inpainted, right_inpainted, left_mask, right_mask


def inpaint_blobs_IR(ir_image):
    """
    Detects and removes bright markers directly from the 16-bit IR image.
    Applies dilation to expand blobs before inpainting.
    Returns the inpainted 16-bit image along with original and processed images with contours overlaid.
    """
    # Compute percentiles to ignore extreme low/high values (for clipping)
    lower_bound = np.percentile(ir_image, 2)
    upper_bound = np.percentile(ir_image, 98)

    if upper_bound <= lower_bound:
        return ir_image.copy(), None, None  # Avoid errors

    # **Step 1: Threshold Directly on the 16-bit IR Image**
    threshold_value = np.percentile(ir_image, 99)  # Use the 99th percentile as the threshold
    binary_mask = np.zeros_like(ir_image, dtype=np.uint8)
    binary_mask[ir_image >= threshold_value] = 255  # Mark bright pixels as 255

    # **Step 2: Find Contours on the 16-bit Binary Mask**
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # **Step 3: Filter Contours by Size**
    minArea = 10  # Minimum blob size to keep
    maxArea = 300  # Maximum blob size to keep
    filtered_contours = [cnt for cnt in contours if minArea <= cv2.contourArea(cnt) <= maxArea]

    # **Step 4: Create a New Mask from Filtered Contours**
    filtered_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filtered_mask, filtered_contours, -1, 255, thickness=-1)

    # **Step 5: Dilate the Mask to Expand the Blobs**
    kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel for dilation
    dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=1)  # Expand blob
    

    # **Step 7: Perform Inpainting Directly in 16-bit**
    inpainted_ir_16bit = cv2.inpaint(ir_image, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    #inpainted_ir_16bit = directional_inpaint(ir_image, dilated_mask, contours)

   
    # **Step 8: Convert IR Images to 8-bit ONLY for Visualization**
    original_8bit = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    processed_8bit = cv2.normalize(inpainted_ir_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert to BGR for Drawing Contours
    processed_image_clipped =clip_IR(processed_8bit,uint_8=True)
    original_contour_image = cv2.cvtColor(original_8bit, cv2.COLOR_GRAY2BGR)
    processed_contour_image = cv2.cvtColor(processed_8bit, cv2.COLOR_GRAY2BGR)

    # **Draw Filtered Contours in RED**
    cv2.drawContours(original_contour_image, filtered_contours, -1, (0, 0, 255), 1)
    cv2.drawContours(processed_contour_image, filtered_contours, -1, (0, 0, 255), 1)

    return inpainted_ir_16bit, original_contour_image, processed_contour_image, processed_image_clipped,dilated_mask

def inpaint_blobs_depth(depth_image, dilated_mask):
    """
    Applies inpainting to the depth image using the mask derived from the IR image.
    The function ensures that only relevant blob regions are inpainted while preserving depth integrity.
    
    :param depth_image: The original 16-bit depth image.
    :param dilated_mask: The mask indicating the regions to be inpainted (from IR processing).
    :return: The inpainted depth image.
    """
    # Ensure the mask is properly formatted
    if dilated_mask is None or np.count_nonzero(dilated_mask) == 0:
        return depth_image.copy()  # No inpainting needed
    
    # Perform inpainting directly on the depth image using the same method as IR
    inpainted_depth_16bit = cv2.inpaint(depth_image, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    return inpainted_depth_16bit

def clip_IR(ir_image, uint_8 = False):
    """
    Clips the IR image by removing extreme intensity values based on percentiles.
    Returns the clipped 16-bit IR image.
    """
    # Compute percentiles to ignore extreme low/high values
    lower_bound = np.percentile(ir_image, 1)
    upper_bound = np.percentile(ir_image, 99)

    if upper_bound <= lower_bound:
        return ir_image.copy()  # Avoid errors
    clipped_ir = np.clip(ir_image, lower_bound, upper_bound)

    if uint_8:
        clipped_ir = cv2.normalize(clipped_ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return clipped_ir
    else:
        # Directly clip in 16-bit range
        return clipped_ir.astype(np.uint16)


def blacken_blobs(ir_image):
    """
    Detects the person's silhouette and marker blobs in an IR image.
    Classifies marker borders as either background or body, then adjusts them accordingly.
    Returns the corrected 16-bit IR image along with original and processed images for visualization.
    """

    # **Step 1: Extract Person's Silhouette (Body Blob)**
    _, body_mask = cv2.threshold(ir_image, np.percentile(ir_image, 40), 255, cv2.THRESH_BINARY)
    body_mask = body_mask.astype(np.uint8)

    # **Step 2: Detect Marker Blobs**
    marker_threshold = np.percentile(ir_image, 99)
    marker_mask = np.zeros_like(ir_image, dtype=np.uint8)
    marker_mask[ir_image >= marker_threshold] = 255  # Mark bright pixels

    # Find contours of markers
    contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # **Step 3: Filter Blobs by Size**
    minArea = 10  
    maxArea = 300  
    filtered_contours = [cnt for cnt in contours if minArea <= cv2.contourArea(cnt) <= maxArea]

    # **Step 4: Create a Mask for Borders & Interiors**
    marker_borders = np.zeros_like(marker_mask)
    marker_interiors = np.zeros_like(marker_mask)

    for cnt in filtered_contours:
        cv2.drawContours(marker_interiors, [cnt], -1, 255, thickness=-1)  # Fill the blob (black interior)
        cv2.drawContours(marker_borders, [cnt], -1, 255, thickness=2)  # Draw borders (white)

    # **Step 5: Combine Body Mask & Marker Borders**
    combined_mask = cv2.bitwise_or(body_mask, marker_borders)

    # **Step 6: Use Canny to Detect Borders**
    canny_edges = cv2.Canny(combined_mask, 100, 200)

    # **Step 7: Classify Borders as 'Body' or 'Background'**
    classified_borders = marker_borders.copy()

    for y, x in zip(*np.where(canny_edges > 0)):  # Iterate over edges
        # If the edge touches the body, color it like the body
        if body_mask[y, x] > 0:
            classified_borders[y, x] = np.median(ir_image[body_mask > 0])  # Match body intensity
        else:
            classified_borders[y, x] = np.median(ir_image[ir_image < np.percentile(ir_image, 10)])  # Match background

    # **Step 8: Merge Adjusted Borders with IR Image**
    adjusted_ir = ir_image.copy()
    adjusted_ir[marker_borders > 0] = classified_borders[marker_borders > 0]

    # **Step 9: Convert IR Images to 8-bit for Visualization**
    original_8bit = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    processed_8bit = cv2.normalize(adjusted_ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert to BGR for Contour Overlay
    processed_image_clipped = clip_IR(processed_8bit, uint_8=True)
    original_contour_image = cv2.cvtColor(original_8bit, cv2.COLOR_GRAY2BGR)
    processed_contour_image = cv2.cvtColor(processed_8bit, cv2.COLOR_GRAY2BGR)

    # **Draw Filtered Contours in RED**
    cv2.drawContours(original_contour_image, filtered_contours, -1, (0, 0, 255), 1)
    cv2.drawContours(processed_contour_image, filtered_contours, -1, (0, 0, 255), 1)

    return adjusted_ir, original_contour_image, processed_contour_image, processed_image_clipped

def visualize_ir_before_after(original_with_contours, processed_with_contours,processed_image_clipped):
	"""
	Displays the original and processed IR images with contours overlaid.
	"""
	# Stack images side by side
	combined = np.hstack((original_with_contours, processed_with_contours))

	# Show images
	cv2.imshow('IR Before (with contours) | After (with contours)', combined)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()

	cv2.imshow('IR clipped After', processed_image_clipped)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()

