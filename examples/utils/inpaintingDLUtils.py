
import json
import os

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from monai.config import KeysCollection
from monai.transforms import (Compose, MapTransform, Resized,
                              ScaleIntensityRanged)
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           find_objects, label)

torch.backends.cudnn.benchmark = True

def load_autoencoder_from_ckpt(ckpt_path: str, device: torch.device) -> AutoencoderKL:
    ae = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=32,
        attention_levels=(False, False, True),
    ).to(device)
    ae.load_state_dict(torch.load(ckpt_path, map_location=device))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae

def load_ddpm_from_ckpt(ckpt_path: str, device: torch.device) -> DiffusionModelUNet:
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=3,             # latent channels
        out_channels=3,
        num_channels=[64, 128, 128],
        attention_levels=[False, True, True],
        num_res_blocks=1,
        num_head_channels=128,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

def display_image(img, title, cmap='gray', figsize=(5, 5), PrintingEnabled=True):
    if PrintingEnabled:
        """
        Displays an image with the specified title.
        If the image has a singleton channel dimension (e.g. shape (1, H, W)),
        it is squeezed to shape (H, W) before displaying.
        
        Parameters:
            img (np.ndarray or torch.Tensor): The image to display.
            title (str): The title for the displayed image.
            cmap (str): The colormap to use (default 'gray').
            figsize (tuple): The figure size.
        """
        # If img is a torch tensor, convert it to a numpy array.
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        # If the image has shape (1, H, W), squeeze the channel dimension.
        if img.ndim == 3 and img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
        
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis("off")
        plt.show(block=False)


def array_to_tensor(image_pixels):
    """
    Converts a numpy image to a torch tensor in channel-first format.
    
    Parameters:
        pil_img (PIL.Image): The input image.
    
    Returns:
        torch.Tensor: The image tensor with shape (1, H, W) and dtype torch.float32.
    """
   
    image_pixels = np.expand_dims(image_pixels, axis=0)  # shape: (1, H, W)
    image_tensor = torch.tensor(image_pixels, dtype=torch.float32)
    return image_tensor

def tensor_to_array(tensor):
    arr = tensor.cpu().numpy()
    if arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    return arr

def make_json_serializable(item):
    if hasattr(item, 'detach'):  # covers torch.Tensor and similar types
        if item.ndim == 0:
            return item.item()
        return item.detach().cpu().tolist()
    elif isinstance(item, dict):
        return {k: make_json_serializable(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [make_json_serializable(i) for i in item]
    return item

def rgbcomparison(base_image, binary_mask):
    """
    Creates an RGB image with:
        - The red channel as the binary mask (scaled to 0-255),
        - The green channel as the base image,
        - The blue channel as zeros.
    
    Parameters:
        base_image (np.ndarray): The input image array (assumed to be in [0,1] or [0,255]).
        binary_mask (np.ndarray): The binary mask (with values 0 or 1).
    
    Returns:
        np.ndarray: The resulting RGB image as a uint8 array.
    """
    if base_image.max() <= 1:
        base_uint8 = (base_image * 255).astype(np.uint8)
    else:
        base_uint8 = base_image.astype(np.uint8)
    if binary_mask.max() <= 1:
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
    else:
        mask_uint8 = binary_mask.astype(np.uint8)
    blue = np.zeros_like(base_uint8, dtype=np.uint8)
    rgb = np.stack([mask_uint8, base_uint8, blue], axis=-1)
    return rgb

class RemoveSmallObjectsTransform(MapTransform):
    def __init__(self, keys, min_size=9):
        super().__init__(keys)
        self.min_size = min_size

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            img_np = img.cpu().numpy()
            for c in range(img_np.shape[0]):
                channel_img = img_np[c]
                labeled_img, num_features = label(channel_img > 0)
                component_sizes = np.bincount(labeled_img.ravel())
                small_objects_mask = np.isin(labeled_img, np.where(component_sizes < self.min_size)[0])
                channel_img[small_objects_mask] = 0
                img_np[c] = channel_img
            data[key] = torch.tensor(img_np, device=img.device)
        return data  

class CropROI(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            original_shape = list(img.shape)  # [channels, height, width]
            non_zero_mask = img.sum(dim=0) > 0
            non_zero_cols = non_zero_mask.sum(dim=0).nonzero(as_tuple=True)[0]
            non_zero_rows = non_zero_mask.sum(dim=1).nonzero(as_tuple=True)[0]
            if len(non_zero_cols) == 0 or len(non_zero_rows) == 0:
                data.setdefault("reversible_info", {})["crop_roi"] = {
                    "original_shape": original_shape,
                    "start_x": None, "start_y": None,
                    "roi_size": None, "end_x": None, "end_y": None,
                }
                continue
            center_x = (non_zero_cols[0] + non_zero_cols[-1]) // 2
            center_y = (non_zero_rows[0] + non_zero_rows[-1]) // 2
            roi_size = 10 + max(non_zero_cols[-1] - non_zero_cols[0] + 1,
                                non_zero_rows[-1] - non_zero_rows[0] + 1)
            start_x = max(0, int(center_x - roi_size // 2))
            start_y = max(0, int(center_y - roi_size // 2))
            end_x = min(img.shape[2], start_x + roi_size)
            end_y = min(img.shape[1], start_y + roi_size)
            cropped_img = img[:, start_y:end_y, start_x:end_x]
            data[key] = cropped_img
            reversible_info = data.get("reversible_info", {})
            reversible_info["crop_roi"] = {
                "original_shape": original_shape,
                "start_x": start_x,
                "start_y": start_y,
                "roi_size": roi_size,
                "end_x": end_x,
                "end_y": end_y,
                "cropped_shape": list(cropped_img.shape),
            }
            data["reversible_info"] = reversible_info
        return data  

class GrayscaleZScoreTransform(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            if img.shape[0] == 3:
                weights = torch.tensor([0.2989, 0.5870, 0.1140],
                                       device=img.device).view(3, 1, 1)
                grayscale = (img * weights).sum(0, keepdim=True)
            elif img.shape[0] == 1:
                grayscale = img
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape[0]}")
            non_zero_values = grayscale[grayscale != 0]
            if non_zero_values.numel() > 0:
                mean = non_zero_values.mean()
                std = non_zero_values.std()
            else:
                mean = torch.tensor(0.0, device=grayscale.device)
                std = torch.tensor(1.0, device=grayscale.device)
            z_score_img = (grayscale - mean) / std
            data[key] = z_score_img
            reversible_info = data.get("reversible_info", {})
            reversible_info["zscore"] = {"mean": mean.item(), "std": std.item()}
            data["reversible_info"] = reversible_info
        return data

class ZeroOutColumnsTransform(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            img = img.permute(1, 2, 0)
            img_np = img.cpu().numpy()
            height = img_np.shape[0]
            top_2_3_rows = img_np[:int(2 * height / 3), :, :]
            non_zero_mask = top_2_3_rows.sum(axis=2).sum(axis=0) > 0
            non_zero_cols = np.nonzero(non_zero_mask)[0]
            if len(non_zero_cols) > 0:
                min_col = non_zero_cols[0]
                max_col = non_zero_cols[-1]
                img_np[:, :min_col, :] = 0
                img_np[:, max_col + 1:, :] = 0
            img = torch.tensor(img_np, device=data[key].device)
            img = img.permute(2, 0, 1)
            data[key] = img
        return data

class ZeroOutRowsTransform(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            img = img.permute(1, 2, 0)
            img_np = img.cpu().numpy()
            rows_sums = np.squeeze(img_np.sum(axis=1))
            img_np[:np.argmin(rows_sums) + 10, :, :] = 0
            img = torch.tensor(img_np, device=data[key].device)
            img = img.permute(2, 0, 1)
            data[key] = img
        return data
    
class CleanBackgroundTrh(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            img_np = img.cpu().numpy()
            img_np = np.where(img_np > 4000, 0, img_np)
            img_np = np.where(img_np < 1000, 0, img_np)
            data[key] = torch.tensor(img_np, device=img.device)
        return data

class CropIRByReversibleInfo(MapTransform):
    def __init__(self, keys, info_key="reversible_info"):
        super().__init__(keys)
        self.info_key = info_key

    def __call__(self, data):
        # Expect data to have both "ir_image" and "reversible_info" keys.
        ir_img = data["ir_image"]
        rev_info = data.get(self.info_key, {})
        crop_info = rev_info.get("crop_roi", {})
        start_x = int(crop_info.get("start_x", 0))
        start_y = int(crop_info.get("start_y", 0))
        end_x = int(crop_info.get("end_x", 0))
        end_y = int(crop_info.get("end_y", 0))
        cropped_ir = ir_img[:, start_y:end_y, start_x:end_x]
        data["ir_image"] = cropped_ir
        return data
def mask_img_to_latent(mask_tensor_img, latent_h, latent_w):
    """
    mask_tensor_img: (1,1,H,W) float {0,1}
    returns: (1,3,latent_h,latent_w) broadcast to 3 channels
    """
    m = F.interpolate(mask_tensor_img, size=(latent_h, latent_w), mode="nearest")
    m3 = m.repeat(1, 3, 1, 1)
    return m3

def process_ir_image(ir_tensor, reversible_info):
    """
    Process an IR image tensor using a Compose pipeline that:
      - Crops the IR image using the ROI coordinates stored in reversible_info,
      - Resizes the resulting crop to 256x256.
    
    Parameters:
        ir_tensor (torch.Tensor): The IR image tensor in channel-first format (1, H, W).
        reversible_info (dict): The reversible transformation information (from depth preprocessing).
    
    Returns:
        ir_roi_array (np.ndarray): The processed IR ROI as a NumPy array.
    """
    data = {"ir_image": ir_tensor, "reversible_info": reversible_info}
    pipeline = Compose([
        CropIRByReversibleInfo(keys=["ir_image"], info_key="reversible_info"),
        Resized(keys=["ir_image"], spatial_size=(256, 256))
    ])
    sample = pipeline(data)
    ir_roi_tensor = sample["ir_image"]
    ir_roi_array = tensor_to_array(ir_roi_tensor)
    return ir_roi_array

def preprocessing_live(image_tensor):
    """
    Process an image tensor through the pipeline and return:
      - The final ROI as a NumPy array (via tensor_to_array), and
      - The reversible transformation information.
      
    This function does not display any intermediate steps.
    
    Parameters:
        image_tensor (torch.Tensor): The input image tensor in channel-first format (1, H, W).
    
    Returns:
        roi_array (np.ndarray): The final ROI as a NumPy array.
        reversible_info (dict): The reversible transformation information.
    """
    data = {"image": image_tensor}
    data["reversible_info"] = {}  # Preallocate reversible info
    
    # Define the processing pipeline.
    pipeline = Compose([
        CleanBackgroundTrh(keys=['image']),
        ZeroOutRowsTransform(keys=['image']),
        RemoveSmallObjectsTransform(keys=['image'], min_size=9),
        ZeroOutColumnsTransform(keys=['image']),
        GrayscaleZScoreTransform(keys=['image']),
        ScaleIntensityRanged(keys=['image'],
                             a_min=-2.0, a_max=+2.0,
                             b_min=0.0, b_max=1.0,
                             clip=True),
        CropROI(keys=['image']),
        Resized(keys=['image'], spatial_size=[256, 256])
    ])
    
    # Run the pipeline.
    sample = pipeline(data)
    
    # Get the final ROI tensor.
    final_roi_tensor = sample["image"]
    
    # Convert the final ROI tensor to a NumPy array.
    roi_array = tensor_to_array(final_roi_tensor)
    
    # Retrieve reversible transformation data.
    reversible_info = sample.get("reversible_info", {})
    
    return roi_array, reversible_info

def clean_image(roi_array, reversible_info):
    """
    Reconstructs the original image from the final ROI and the reversible transformation information.
    
    Parameters:
        roi_array (np.ndarray): The final ROI as produced by preprocessing_live (values in [0,1]).
        reversible_info (dict): The reversible transformation information produced during preprocessing.
    
    Returns:
        reconstructed_clipped (np.ndarray): The final reconstructed image as a uint16 NumPy array.
    """
    # Extract ROI coordinates and original image shape from reversible_info.
    crop_info = reversible_info.get("crop_roi", {})
    start_x = int(crop_info.get("start_x", 0))
    start_y = int(crop_info.get("start_y", 0))
    end_x = int(crop_info.get("end_x", 0))
    end_y = int(crop_info.get("end_y", 0))
    roi_width = end_x - start_x
    roi_height = end_y - start_y
    
    # Step 1: Resize the ROI produced by preprocessing_live back to the original ROI dimensions.
    # (roi_array is assumed to be in the [0,1] range.)
    roi_img = Image.fromarray((roi_array * 255).astype(np.uint8))
    roi_img = roi_img.resize((roi_width, roi_height), resample=Image.BICUBIC)
    roi_array_resized = np.array(roi_img).astype(np.float32) / 255.0
    
    # Step 2: Reverse intensity scaling.
    # Original mapping (assumed): scaled = (original_z + 2) / 4, so inverse: original_z = scaled*4 - 2.
    z_value = roi_array_resized * 4.0 - 2.0
    
    # Step 3: Reverse z-score normalization.
    zscore_info = reversible_info.get("zscore", {})
    mean = float(zscore_info.get("mean", 0))
    std = float(zscore_info.get("std", 1))
    # Modified logic: if z_value <= -2, set to 0; else, original = z_value*std + mean.
    original_roi = np.where(z_value <= -2, 0, z_value * std + mean)
    
    # Step 4: Reconstruct the original image using the original shape.
    original_shape = crop_info.get("original_shape", [1, 256, 256])
    orig_height = int(original_shape[1])
    orig_width = int(original_shape[2])
    
    # Create a blank canvas.
    reconstructed = np.zeros((orig_height, orig_width), dtype=np.float32)
    
    # Ensure ROI fits within original dimensions.
    if end_x > orig_width or end_y > orig_height:
        raise ValueError("ROI placement exceeds original dimensions.")
    
    reconstructed[start_y:end_y, start_x:end_x] = original_roi
    
    # (Flipping is no longer performed.)
    
    # Step 5: Clip intensities to [0, 65535] and convert to uint16.
    reconstructed_clipped = np.clip(reconstructed, 0, 65535).astype(np.uint16)
 
    
    return reconstructed_clipped

def create_binary_mask(image_array, lower_percentile=0, upper_percentile=99):
    """
    Creates a binary mask from an image array such that all nonzero pixels
    with values between the given lower and upper percentiles (computed from the nonzero pixels)
    are set to 1, and all other pixels are set to 0.
    
    Parameters:
        image_array (np.ndarray): The input image array.
        lower_percentile (float): The lower percentile threshold (default is 0).
        upper_percentile (float): The upper percentile threshold (default is 99).
        
    Returns:
        np.ndarray: A binary mask (dtype=np.uint8) with the same shape as image_array.
    """
    non_zero_pixels = image_array[image_array > 0]
    if non_zero_pixels.size == 0:
        return np.zeros_like(image_array, dtype=np.uint8)
    
    lower_threshold = np.percentile(non_zero_pixels, lower_percentile)
    upper_threshold = np.percentile(non_zero_pixels, upper_percentile)
    
    binary_mask = np.where((image_array >= lower_threshold) & (image_array < upper_threshold), 1, 0).astype(np.uint8)
    return binary_mask

def detect_blobs(binary_mask):
    """
    Performs blob detection on a binary mask by labeling connected components.

    Parameters:
        binary_mask (np.ndarray): A binary mask (with values 0 and 1).

    Returns:
        labeled_mask (np.ndarray): An array with the same shape as binary_mask, where each connected component
                                   (blob) is assigned a unique label (0 is the background).
        num_blobs (int): The number of detected blobs.
        blob_slices (list of slice tuples): A list of slice objects corresponding to the bounding box of each blob.
    """
    # Label connected components in the binary mask.
    labeled_mask, num_blobs = label(binary_mask)
    
    # Optionally, use find_objects to get slices (bounding boxes) for each blob.
    blob_slices = find_objects(labeled_mask)
    
    return labeled_mask, num_blobs, blob_slices

def create_color_labeled_image(labeled_mask):
    """
    Creates a color image from a labeled mask.
    Each unique label (except 0) is mapped to a distinct color using a colormap.
    Background (label 0) is set to black.
    
    Parameters:
        labeled_mask (np.ndarray): A 2D array with labels (0 is background).
    
    Returns:
        np.ndarray: A color (RGB) image as a uint8 array.
    """
    # Get the number of labels; we add one so that label values map correctly.
    num_labels = np.max(labeled_mask) + 1
    # Create a colormap (we use HSV so that hues vary).
    cmap = cm.get_cmap('hsv', num_labels)
    
    # Normalize the labels to the range [0, 1] for the colormap.
    norm = colors.Normalize(vmin=0, vmax=num_labels-1)
    
    # Map each label to its corresponding color.
    colored = cmap(norm(labeled_mask))
    # colored is an MxNx4 array (RGBA) in float; set background (label 0) to black:
    colored[labeled_mask == 0] = [0, 0, 0, 1]
    # Convert to uint8 and drop the alpha channel.
    rgb_image = (colored[..., :3] * 255).astype(np.uint8)
    return rgb_image
def expand_blob_perimeter(binary_mask, n_steps=1):
    """
    Expands only the perimeter of the blobs in a binary mask.

    Parameters:
        binary_mask (np.ndarray): Input binary mask (0s and 1s).
        n_steps (int): Number of dilation iterations to expand the perimeter.

    Returns:
        np.ndarray: Updated binary mask with the expanded blob perimeters.
    """
    # Step 1: Extract the perimeter of the blobs.
    # Erode the mask so that the boundaries shrink.
    eroded_mask = binary_erosion(binary_mask)
    # The difference between the original mask and its erosion is the perimeter.
    blob_perimeter = binary_mask & (~eroded_mask)
    
    # Step 2: Dilate only the perimeter.
    expanded_perimeter = binary_dilation(blob_perimeter, iterations=n_steps)
    
    # Step 3: Combine the dilated perimeter with the original mask.
    updated_mask = binary_mask | expanded_perimeter
    
    return updated_mask.astype(np.uint8)

def expand_blobs_with_conditions(binary_mask, binary_mask2, roi_with_holes1, num_iterations):
    """
    Expands blobs in 'binary_mask' based on the distance transform over a specified number of iterations.
    
    For each iteration:
      - Computes the distance transform of the blobs so that edge pixels (those adjacent to background in the
        8-nearest neighborhood) have a distance of 1 and inner pixels have larger distances.
      - Increases nonzero distances by 1 so that edge pixels have a value of 2.
      - Expands the blobs by setting pixels to 1 if they have an 8-connected neighbor with a value of 2,
        provided that:
          * The corresponding pixel in 'roi_with_holes1' is not 0, and
          * The corresponding pixel in 'binary_mask2' is not 1.
      - After expansion, all nonzero values in the updated mask are thresholded to 1.
    
    Parameters:
        binary_mask (np.ndarray): Binary mask (values 0 or 1) representing the blobs.
        binary_mask2 (np.ndarray): Secondary binary mask used as a condition (values 0 or 1).
        roi_with_holes1 (np.ndarray): ROI image array where pixels with value 0 are ineligible for expansion.
        num_iterations (int): Number of iterations to perform (will be cast to np.uint8).
    
    Returns:
        np.ndarray: The updated binary mask after performing blob expansion.
    """
    num_iterations = np.uint8(num_iterations)
    for _ in range(int(num_iterations)):
        # Compute the distance transform: foreground pixels (value 1) get a distance to the nearest background pixel.
        dt = distance_transform_edt(binary_mask)
        # Floor the distances so that edge pixels (adjacent to background) become 1 and inner pixels >1.
        dt_int = np.floor(dt).astype(np.uint8)
        # Increase nonzero distances by 1 so that edge pixels now have value 2.
        dt_plus_one = dt_int.copy()
        dt_plus_one[dt_int > 0] += 1

        # Identify edge pixels (those with a value of 2 after the increment).
        edge_pixels = (dt_plus_one == 2)
        # Dilate the edge pixels with an 8-connected structure to find neighbors.
        structure = np.ones((3, 3), dtype=np.uint8)
        dilated_edges = binary_dilation(edge_pixels, structure=structure)

        # Candidate pixels for expansion:
        #   - They are not already part of the blob (binary_mask is 0).
        #   - They have an 8-neighbor that is an edge pixel.
        #   - The corresponding pixel in roi_with_holes1 is not 0.
        #   - The corresponding pixel in binary_mask2 is not 1.
        candidate = (
            (~binary_mask.astype(bool)) &
            dilated_edges &
            (roi_with_holes1 != 0) &
            (binary_mask2 != 1)
        )

        # Expand the blob: set candidate pixels to 1.
        binary_mask[candidate] = 1

        # Ensure the mask remains binary.
        binary_mask = (binary_mask > 0).astype(np.uint8)
    
    return binary_mask

def inpaint_single_image(image_array,
                         mask_array,
                         ae: AutoencoderKL,
                         model: DiffusionModelUNet,
                         num_inference_steps: int = 1000,
                         num_resample_steps: int = 1,
                         n_resamples: int = 3,
                         device: torch.device = None):
    print(f"inpainting on device: {device} | n_resamples={n_resamples}")

    display_image(image_array, "Original Image")

    # mask_array expected: 1 where masked in your pipeline -> invert to 1 = KNOWN
    mask_array = np.logical_not(mask_array).astype(np.uint8)
    display_image(mask_array, "Inverted Mask (1 = known)")

    if device is None:
        device = next(model.parameters()).device

    # Prepare tensors WITHOUT resizing
    if image_array.ndim == 2:
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif image_array.ndim == 3:
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)               # (1,C,H,W)
    else:
        raise ValueError("Unsupported image_array shape. Expected 2D or 3D.")

    if mask_array.ndim == 2:
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)    # (1,1,H,W)
    elif mask_array.ndim == 3:
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError("Unsupported mask_array shape. Expected 2D or 3D.")

    original_size = image_tensor.shape[-2:]
    image_tensor = image_tensor.to(device)
    mask_tensor  = mask_tensor.to(device)

    # Encode once
    with torch.no_grad():
        z, _ = ae.encode(image_tensor)      # (1,3,H_l,W_l)

    H_l, W_l = z.shape[-2], z.shape[-1]
    mask_latent = mask_img_to_latent(mask_tensor, H_l, W_l)  # (1,3,H_l,W_l), 1 = known

    # Prepare scheduler (we'll reset timesteps each resample for safety)
    scheduler = DDPMScheduler(num_train_timesteps=max(1000, int(num_inference_steps)))

    decoded_accum = None

    for r in range(int(n_resamples)):
        # Fresh noise run
        scheduler.set_timesteps(int(num_inference_steps))
        known_lat = z * mask_latent
        x = torch.randn_like(known_lat)

        with torch.no_grad():
            for t in scheduler.timesteps:
                for _ in range(num_resample_steps):
                    if t > 0:
                        noise = torch.randn_like(x)
                        known_noised = scheduler.add_noise(known_lat, noise=noise, timesteps=(t - 1))
                        t_vec = torch.full((x.size(0),), t, device=device, dtype=torch.long)
                        pred = model(x, timesteps=t_vec)
                        step_out = scheduler.step(pred, t, x)
                        x_prev = step_out[0] if isinstance(step_out, (tuple, list)) else step_out.prev_sample
                        x = torch.where(mask_latent == 1, known_noised, x_prev)

        # Ensure known latents are exact
        x = torch.where(mask_latent == 1, z, x)

        # Decode this resample
        with torch.no_grad():
            decoded = ae.decode(x)          # (1,1,H,W)
        decoded = torch.clamp(decoded, 0, 1)

        # Accumulate
        if decoded_accum is None:
            decoded_accum = decoded.float()
        else:
            decoded_accum = decoded_accum + decoded.float()

    # Mean over resamples
    decoded_mean = decoded_accum / float(n_resamples)
    print("Decoded mean shape:", decoded_mean.shape)
    display_image(decoded_mean.squeeze(0), f"Inpainted mean of {n_resamples} resamples")

    # To numpy
    result = decoded_mean.cpu().squeeze(0).numpy()

    # Compose: keep known pixels from original (no thresholding)
    if result.ndim == 3 and result.shape[0] == 1:
        result2d = result[0]
    elif result.ndim == 2:
        result2d = result
    else:
        raise ValueError("Unexpected decoded shape; expected (1,H,W) or (H,W).")

    if image_array.ndim == 3 and image_array.shape[0] == 1:
        img2d = image_array[0]
    elif image_array.ndim == 2:
        img2d = image_array
    else:
        raise ValueError("Unexpected image_array shape; expected (H,W) or (1,H,W).")

    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        m2d = mask_array[0]
    elif mask_array.ndim == 2:
        m2d = mask_array
    else:
        raise ValueError("Unexpected mask_array shape; expected (H,W) or (1,H,W).")

    # m2d == 1 -> known pixels from original
    result = np.where(m2d, img2d, result2d)
    result = np.clip(result, 0, 1)

    return result