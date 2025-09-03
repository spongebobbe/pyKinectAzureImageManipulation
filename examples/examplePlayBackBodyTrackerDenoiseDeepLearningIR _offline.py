import os
import cv2
import json

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label,binary_dilation,binary_erosion

import re
import matplotlib.pyplot as plt
import time
import tqdm

from examples.utils.azureKinectFileUtils import makeC3d
from examples.utils.inpaintingUtils import inpaint_blobs_IR, inpaint_blobs_depth, visualize_ir_before_after
from examples.utils.azureKinectFileUtils import BONE_LIST, extract_skeleton_data, makeC3d 
from examples.utils.inpaintingDLUtils import (
	array_to_tensor,
	clean_image,
	create_binary_mask,
	create_color_labeled_image,
	detect_blobs,
	display_image,
	expand_blob_perimeter,
	expand_blobs_with_conditions,
	inpaint_single_image,
	load_autoencoder_from_ckpt,
	load_ddpm_from_ckpt,
	make_json_serializable,
	mask_img_to_latent,
	preprocessing_live,
	process_ir_image,
	rgbcomparison
)




def main():
    # ---- paths ----
    # Root folder
    ROOT = r"C:\Users\Feder\sources\kinect_thesis\test_pipeline"
    # depth_images_path = r"C:\Users\Feder\sources\kinect_thesis\test_pipeline\_Backup_TestSubjects\Depth\LAURA_TESTSUBJECT\VICONKINECT\HABL"
    # ir_images_path    = r"C:\Users\Feder\sources\kinect_thesis\test_pipeline\_Backup_TestSubjects\IR\LAURA_TESTSUBJECT\VICONKINECT\HABL"
    # save_path         = r"C:\Users\Feder\sources\kinect_thesis\test_pipeline\Output_Inpainted_images"


    # Change these two when needed
    TEST_SUBJECT = "LAURA_TESTSUBJECT"
    EXERCISE     = "HABL"

    # Usually constant, but editable if needed
    SYSTEM = "VICONKINECT"

    depth_images_path = os.path.join(ROOT, "_Backup_TestSubjects", "Depth", TEST_SUBJECT, SYSTEM, EXERCISE)
    ir_images_path    = os.path.join(ROOT, "_Backup_TestSubjects", "IR",    TEST_SUBJECT, SYSTEM, EXERCISE)
    save_path         = os.path.join(ROOT, "_Backup_TestSubjects", "Output_Inpainted_images", TEST_SUBJECT, SYSTEM, EXERCISE)

    os.makedirs(save_path, exist_ok=True)

    # ---- device & models ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    AE_CKPT  = r"C:\Users\Feder\sources\kinect_thesis\autoencoder_training_run_2025_05_09_at_19_41_45_810_FFT\best_autoencoder_saved\autoencoder_training_run_2025_05_09_at_19_41_45_810_FFT_best_model_epoch_205.pth"
    DDPM_CKPT = r"C:\Users\Feder\sources\kinect_thesis\diffusion_training_run_2025_05_18_at_19_03_00_550_autoencoder\best_model\diffusion_training_run_2025_05_18_at_19_03_00_550_autoencoder_best_epoch_535.pth"
    ae  = load_autoencoder_from_ckpt(AE_CKPT, device)
    ddpm = load_ddpm_from_ckpt(DDPM_CKPT, device)
    print("Models loaded.")

    # Turn off interactive plotting to avoid popping windows during batch
    plt.ioff()

    # ---- gather depth frames, sort by frame number ----
    depth_re = re.compile(r"^depth(\d+)\.png$", re.IGNORECASE)
    def frame_num(name):
        m = depth_re.match(name)
        return int(m.group(1)) if m else None

    depth_files = [
        f for f in os.listdir(depth_images_path)
        if depth_re.match(f)
    ]
    depth_files.sort(key=frame_num)
    print(f"Found {len(depth_files)} depth frames.")

    if not depth_files:
        print(f"No depth frames found in: {depth_images_path}")
        return

    # ---- parameters for inpainting ----
    n_resampling  = 3
    n_inference   = 200

    total = len(depth_files)
    done  = 0

    for df in tqdm(depth_files, desc="Processing frames"):
        m = depth_re.match(df)
        frame = m.group(1)
        depth_path = os.path.join(depth_images_path, df)
        #print(f"\n processing {df} {done}/{total} (<{done/total*100:.2f}%>)")
        ir_file    = f"ir{frame}.png"
        ir_path    = os.path.join(ir_images_path, ir_file)

        if not os.path.exists(ir_path):
            print(f"[WARN] Missing IR for frame {frame}: {ir_path} — skipping.")
            continue

        try:
            # --- DEPTH preprocess ---
            depth_image  = Image.open(depth_path)
            depth_tensor = array_to_tensor(depth_image)
            roi_array, rev_info = preprocessing_live(depth_tensor)
            #reconstructed_depth = clean_image(roi_array, rev_info)  # (not saved; used to map back later)

            # --- IR preprocess (cropped using depth ROI info) ---
            ir_image  = Image.open(ir_path)
            ir_tensor = array_to_tensor(ir_image)
            ir_roi_array = process_ir_image(ir_tensor, rev_info)

            # --- masks ---
            binary_mask  = create_binary_mask(ir_roi_array, lower_percentile=99, upper_percentile=100)
            binary_mask2 = create_binary_mask(ir_roi_array, lower_percentile=90, upper_percentile=99)
            binary_mask2 = binary_erosion(binary_mask2, iterations=1)
            binary_mask3 = expand_blobs_with_conditions(binary_mask, binary_mask2, roi_array, 4)

            # --- inpaint on the 256x256 ROI ---
            inpainted_roi = inpaint_single_image(
                roi_array,
                binary_mask3,
                ae=ae,
                model=ddpm,
                num_inference_steps=n_inference,
                num_resample_steps=n_resampling,
                device=device,
            )

            # --- map back to original size & save ---
            reconstructed_inpainted = clean_image(inpainted_roi, rev_info)
            out_uint16 = reconstructed_inpainted.astype(np.uint16)
            out_name = f"depth_inpainted{frame}.png"
            out_path = os.path.join(save_path, out_name)
            Image.fromarray(out_uint16).save(out_path, format="PNG")

            done += 1
            #print(f"[OK] {frame}: saved -> {out_path}")

            # Close any figures that might have been created inside helper functions
            plt.close('all')

        except Exception as e:
            print(f"[ERROR] frame {frame}: {e}")

    print(f"Finished. Processed {done}/{total} (<{done/total*100:.2f}%>) depth frames.")
    # ---- OPTIONAL: display various images for the last processed frame ----

	# display_image(roi_array, "Final ROI (Depth)")
	# display_image(reconstructed_image, "Reconstructed Image (Depth)")


	# display_image(binary_mask, "IR Binary Mask")
	# display_image(binary_mask2, "IR Binary Mask2")
	# display_image(binary_mask3, "Depth Binary Mask")


	# display_image(rgb_ir_image, "RGB IR Image (Red: Mask, Green: IR, Blue: 0)")
	# display_image(rgb_roi_image, "RGB ROI Image (Red: Mask, Green: ROI, Blue: 0)")


	# display_image(rgb_labeled_image, "Labeled Blobs (Color Coded)")


	# display_image(inpainted_image, "Inpainted Image (Depth)")
	# display_image(reconstructed_impainted_image, "Reconstructed Inpainted Image (Depth)")

if __name__ == "__main__":
	main()
	plt.show()  # Keep plots open at the end of the script

if __name__ == "__main__":
	main()
