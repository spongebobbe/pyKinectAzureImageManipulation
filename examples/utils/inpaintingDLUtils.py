import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from monai.config import KeysCollection
from monai.transforms import Compose, Resized, ScaleIntensityRanged, MapTransform
from scipy.ndimage import label,binary_dilation,binary_erosion,distance_transform_edt,binary_dilation, label, find_objects
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
import torch.nn.functional as F



import json

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

def tensor_to_array(tensor):
    arr = tensor.cpu().numpy()
    if arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    return arr 

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


def display_image(img, title, cmap='gray', figsize=(5, 5)):
        if False:

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
            plt.show(block=True)


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


def preprocess_ir_image(ir_tensor, reversible_info):
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


def inpaint_single_image(image_array, mask_array, config_path, model_path, num_inference_steps=None, num_resample_steps=1, device=None):
    """
    Loads the configuration and model, resizes the input image and mask according to the spatial dimensions
    specified in the configuration, then performs inpainting on the image using the provided mask.
    The inpainting is carried out based on the known (unmasked) and unknown (masked) regions without further preprocessing.
    Inference is performed under torch.no_grad(). The final output is resized back to the original input image size,
    and the known pixels from the original (non-resized) image are substituted back into the result.
    
    Parameters:
        image_array (np.ndarray): Input image array (ROI) to be inpainted.
            Expected shape: (H, W) or (C, H, W) where C is typically 1.
        mask_array (np.ndarray): Binary mask array to guide the inpainting.
            Expected shape: (H, W) or (C, H, W) matching the image spatial dimensions.
        config_path (str): Path to the JSON configuration file.
        model_path (str): Path to the model weights file.
        num_inference_steps (int, optional): Number of timesteps for inference. If None, it will be taken from the config.
        num_resample_steps (int, optional): Number of resampling steps per timestep (default: 1).
        device (torch.device, optional): Device to run inference on. Defaults to CUDA if available, else CPU.
    
    Returns:
        np.ndarray: The inpainted image as a NumPy array (converted to channel-last format)
                    resized back to the original input image size, with known pixels substituted from the original image.
    """

    display_image(image_array, "Original Image")
    # Invert the binary mask (so that 1 corresponds to known pixels)
    mask_array = np.logical_not(mask_array)
    display_image(mask_array, "Inverted Mask")
    
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
        # print("Configuration:")
        # print(json.dumps(config, indent=4))
    
    # Set default number of inference steps if not provided
    if num_inference_steps is None:
        num_inference_steps = config.get("scheduler_steps", 200)
    
    # Load the model
    model = DiffusionModelUNet(
        spatial_dims=config["spatial_dims"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        num_channels=config["model_channels"],
        attention_levels=config["attention_levels"],
        num_res_blocks=config["num_res_blocks"],
        num_head_channels=config["num_head_channels"]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Create scheduler and set timesteps
    scheduler = DDPMScheduler(num_train_timesteps=num_inference_steps)
    scheduler.set_timesteps(num_inference_steps)
    # print("Number of inference steps:", num_inference_steps)
    
    # Convert image_array to a torch tensor with shape (1, C, H, W)
    if image_array.ndim == 2:
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    elif image_array.ndim == 3:
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError("Unsupported image_array shape. Expected a 2D or 3D array.")
    
    # Store original spatial size (H, W) for later restoration
    original_size = image_tensor.shape[-2:]
    
    # Convert mask_array to a torch tensor with shape (1, C, H, W)
    if mask_array is None:
        raise ValueError("mask_array must be provided.")
    if mask_array.ndim == 2:
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    elif mask_array.ndim == 3:
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError("Unsupported mask_array shape. Expected a 2D or 3D array.")
    
    # Resize the image and mask according to the spatial dimensions specified in the config
    spatial_size = config.get("spatial_size")
    if spatial_size is not None:
        image_tensor_resized = F.interpolate(image_tensor, size=spatial_size, mode='bilinear', align_corners=False)
        mask_tensor_resized = F.interpolate(mask_tensor, size=spatial_size, mode='nearest')
    else:
        image_tensor_resized = image_tensor
        mask_tensor_resized = mask_tensor
    
    # (Optional) Display resized image and mask
    display_image(image_tensor_resized.squeeze(0), "Resized Image")
    display_image(mask_tensor_resized.squeeze(0), "Resized Mask")
    # print("Resized image and mask shapes:")
    # print(image_tensor_resized.shape)
    # print(mask_tensor_resized.shape)
    
    # Move tensors to the correct device
    image_tensor_resized = image_tensor_resized.to(device)
    mask_tensor_resized = mask_tensor_resized.to(device)
    
    # Prepare the masked image and initialize the inpainted image with random noise
    masked_image = image_tensor_resized * mask_tensor_resized
    inpainted_image = torch.randn_like(masked_image).to(device)
    
    # Perform inpainting inference without gradient computation
    with torch.no_grad():
        for t in scheduler.timesteps:
            for _ in range(num_resample_steps):
                noise = torch.randn_like(inpainted_image).to(device)
                if t > 0:
                    known_part = scheduler.add_noise(
                        original_samples=masked_image, noise=noise, timesteps=(t - 1)
                    )
                    t_tensor = torch.full((inpainted_image.size(0),), t, device=device, dtype=torch.long)
                    model_output = model(inpainted_image, timesteps=t_tensor)
                    unknown_part, _ = scheduler.step(model_output, t, inpainted_image)
                    # Combine known (unmasked) and unknown (masked) regions
                    inpainted_image = torch.where(mask_tensor_resized == 1, known_part, unknown_part)
    
    # Clip the final image to [0, 1] and ensure known pixels remain unchanged in the resized domain
    inpainted_image = torch.clamp(inpainted_image, 0, 1)
    inpainted_image = torch.where(mask_tensor_resized == 1, image_tensor_resized, inpainted_image)
    
    # Remove the batch dimension from the inpainted image
    result = inpainted_image.cpu().squeeze(0)
    display_image(result, "Inpainted Image low res")

    # Resize the result back to the original input image size
    result = F.interpolate(result.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
    display_image(result, "original dimensions Inpainted Image")
    result = result.numpy()
    
    # Substitute the known pixels from the original image back into the result
    result = np.where(result <= np.percentile(result[result > 0], 25), 0, result)
    result = np.where(mask_array, image_array, result)
    
    # Convert result from channel-first (C, H, W) to channel-last (H, W, C) format:
    if result.ndim == 3:
        if result.shape[0] == 1:
            # For single channel images, remove the channel dimension
            result = result[0]
        else:
            # For multi-channel images, transpose the dimensions
            result = np.transpose(result, (1, 2, 0))
    
    return result

