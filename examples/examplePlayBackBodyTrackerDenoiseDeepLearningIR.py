import os
import cv2
import json
import pykinect_azure as pykinect

from examples.utils.azureKinectFileUtils import makeC3d
from examples.utils.inpaintingUtils import inpaint_blobs_IR, inpaint_blobs_depth, visualize_ir_before_after
from examples.utils.azureKinectFileUtils import BONE_LIST, extract_skeleton_data, makeC3d 
from examples.utils.inpaintingDLUtils import array_to_tensor, clean_image, create_binary_mask, display_image, expand_blobs_with_conditions, inpaint_single_image, preprocess_ir_image, preprocessing_live 

from pykinect_azure.k4arecord.playback import Playback
from scipy.ndimage import binary_erosion


if __name__ == "__main__":
	plot = True  # Set to True to enable plotting
	modify_depth = True  # Set to True to modify depth image
	modify_IR = False
	modify_RGB = False  # not implemented
	video_filename = 'D:\mkv_recordings\VICON KINECT\\06_9 - 11 LAURA TESTSUBJECT\VICONKINECT\SAB.mkv'
	output_json = 'laura_SAB.json' 

	# Initialize the library
	pykinect.initialize_libraries(track_body=True)

	# Start modified playback
	playback = Playback(video_filename)
	calibration = playback.get_calibration()

	# Start body tracker
	bodyTracker = pykinect.start_body_tracker(calibration=calibration)
	
	all_frames = []  # Store skeleton data for all frames

	frame_count = 0
	while True:
		# Get camera capture
		ret, capture = playback.update()
		if not ret:
			break

		# Modify the depth image BEFORE skeletal tracking (if flag is set)
		if modify_depth and not(modify_IR):
			ret_depth, depth_image = capture.get_depth_image()
			if ret_depth and depth_image is not None:
				#preprocess image before model
				depth_tensor = array_to_tensor(depth_image)
				roi_array, rev_info = preprocessing_live(depth_tensor)
				print(f"Original tensor image shape: {depth_tensor.shape}")
				#display_image(roi_array, "Final ROI (Depth)")
				#print("Reversible Transformation Info:")
				#print(json.dumps(make_json_serializable(rev_info), indent=4))
				#----------------------------------
				reconstructed_image = clean_image(roi_array, rev_info)
				print("Reconstructed Image (Depth)")
				print(reconstructed_image.shape)
				#display_image(reconstructed_image, "Reconstructed Image (Depth)")
				 #----------------------------------
				ret_ir, IR_image = capture.get_ir_image()
				ir_tensor = array_to_tensor(IR_image)
				ir_roi_array = preprocess_ir_image(ir_tensor, rev_info)
				#display_image(ir_roi_array, "Processed IR ROI")
				#----------------------------------
				# Create a binary mask from the IR ROI.
				binary_mask = create_binary_mask(ir_roi_array, lower_percentile=99, upper_percentile=100)
				#dilated = binary_dilation(binary_mask, iterations=10)
				display_image(binary_mask, "IR Binary Mask")

				binary_mask2=create_binary_mask(ir_roi_array, lower_percentile=90, upper_percentile=99)
				binary_mask2=binary_erosion(binary_mask2,iterations=2)
				display_image(binary_mask2, "IR Binary Mask2")

				binary_mask3= expand_blobs_with_conditions(binary_mask, binary_mask2, roi_array, 4)
				display_image(binary_mask3, "Depth Binary Mask")

				'''
				# Create an RGB image using the IR ROI and the binary mask.
				rgb_ir_image = rgbcomparison(ir_roi_array, binary_mask)
				display_image(rgb_ir_image, "RGB IR Image (Red: Mask, Green: IR, Blue: 0)")
				
				# Also create an RGB image using the depth ROI and the binary mask.
				rgb_roi_image = rgbcomparison(roi_array, binary_mask3)
				display_image(rgb_roi_image, "RGB ROI Image (Red: Mask, Green: ROI, Blue: 0)")

				# Create a single color image using the labeled mask.
				rgb_labeled_image = create_color_labeled_image(labeled_mask)
				display_image(rgb_labeled_image, "Labeled Blobs (Color Coded)")
				
				#roi_with_holes1=np.where(binary_mask == 0, roi_array, 0)
				# display_image(roi_with_holes1, "ROI with Holes1")
				#roi_with_holes3=np.where(binary_mask3 == 0, roi_array, 0)
				# display_image(roi_with_holes3, "ROI with Holes3")
				
				'''


				#----------------------------------
				run_dir=r"D:\mkv_recordings\VICON KINECT\federico_model\run_2024_12_09_at_13_54_33"
				run_name=r"run_2024_12_09_at_13_54_33"
				model_folder=r"D:\mkv_recordings\VICON KINECT\federico_model\run_2024_12_09_at_13_54_33\best_model_saved"
				model_name=r"run_2024_12_09_at_13_54_33_best_model_epoch_480.pth"

				config_path = os.path.join(run_dir, "best_model_parameters", f"{run_name}_config.json")
				model_path = os.path.join(model_folder, model_name)

				inpainted_image = inpaint_single_image(roi_array, binary_mask3, config_path, model_path, num_inference_steps=1000, num_resample_steps=1)
				display_image(inpainted_image, "Inpainted Image (Depth)")
				print(inpainted_image.shape)

				reconstructed_impainted_image = clean_image(inpainted_image, rev_info)
				
				display_image(reconstructed_impainted_image, "Reconstructed Inpainted Image (Depth)")


				


				playback._set_depth_image_to_capture(reconstructed_impainted_image)
			
		if modify_RGB:
			pass
						
		ret_ir, IR_image = capture.get_ir_image()
		#ir_before_clipping = clip_IR(IR_image, uint_8=True)
		#cv2.imshow('Infrared Image before clipped',ir_before_clipping)
		if modify_IR and not(modify_depth):
			ret_ir, IR_image = capture.get_ir_image()
			if ret_ir and IR_image is not None:
				modified_ir_image, original_contours, processed_contours, processed_image_clipped, blob_mask = inpaint_blobs_IR(IR_image)
			
			#modified_ir_image = np.zeros_like(IR_image, dtype=np.uint16)
			#modified_ir_image = np.ascontiguousarray(np.fliplr(IR_image))

			playback._set_ir_image_to_capture(modified_ir_image)
			visualize_ir_before_after(original_contours, processed_contours, processed_image_clipped)
			# Plot image
			
		
		if (modify_depth and modify_IR): 
			ret_depth, depth_image = capture.get_depth_image()
			if ret_depth and depth_image is not None:
				modified_depth_image = inpaint_blobs_depth(depth_image,blob_mask)
				playback._set_depth_image_to_capture(modified_depth_image)

		
		#ret_ir, IR_image = capture.get_ir_image()
		#ir_after_clipping = clip_IR(IR_image, uint_8=True)
		#cv2.imshow('Infrared Image after clipped', ir_after_clipping)
		
		ret_depth, depth_color_image = capture.get_colored_depth_image()
		ret_color, color_image_transformed = capture.get_transformed_color_image()
		if not ret_depth or not ret_color:
			continue
	 #cv2.imshow("Depth", depth_color_image)

		# Run skeletal tracking
		body_frame = bodyTracker.update(capture=capture)
		ret_seg, body_image_color = body_frame.get_segmentation_image()

		if not ret_seg:
			continue
		
		 # Get timestamp
		timestamp_usec = body_frame.get_device_timestamp_usec()

		frame_skeleton_data = extract_skeleton_data(body_frame)
		all_frames.append({
			"bodies": frame_skeleton_data,
			"frame_id": frame_count,
			"num_bodies": len(frame_skeleton_data),
			"timestamp_usec": timestamp_usec
		})
		frame_count += 1

		
		combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
		combined_image = cv2.addWeighted(color_image_transformed[:, :, :3], 0.7, combined_image, 0.3, 0)
		combined_image = body_frame.draw_bodies(combined_image)
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 1
		font_color = (0, 255, 0)  # Green color
		thickness = 2
		position = (50, 50)  # Top-left corner

		# Overlay frame count on the image
		cv2.putText(combined_image, f'Frame: {frame_count}', position, font, font_scale, font_color, thickness, cv2.LINE_AA)
		if plot:
			# Display the image
			cv2.imshow('Depth with Skeleton', combined_image) 
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		output_dir = "saved_images"

		'''
		if frame_count ==372:
			cv2.imwrite(os.path.join(output_dir, "infrared_before_clipped.png"), ir_before_clipping)
			cv2.imwrite(os.path.join(output_dir, "infrared_after_clipped.png"), ir_after_clipping)
			cv2.imwrite(os.path.join(output_dir, "depth_with_skeleton.png"), combined_image)
			print(f"Saved images at frame {frame_count}")
		'''

		if frame_count ==10:
			cv2.imwrite(os.path.join(output_dir, "depth_with_skeleton_laura10.png"), combined_image)
			print(f"Saved images at frame {frame_count}")
		if frame_count ==50:
			cv2.imwrite(os.path.join(output_dir, "depth_with_skeleton_laura50.png"), combined_image)
			print(f"Saved images at frame {frame_count}")
		if frame_count ==100:
			cv2.imwrite(os.path.join(output_dir, "depth_with_skeleton_laura100.png"), combined_image)
			print(f"Saved images at frame {frame_count}")
		if frame_count ==200:
			cv2.imwrite(os.path.join(output_dir, "depth_with_skeleton_laura100.png"), combined_image)
			print(f"Saved images at frame {frame_count}")


	if plot:
		cv2.destroyAllWindows()

	# Save skeleton data in the correct JSON structure
	json_output = {
		"bone_list": BONE_LIST,
		"frames": all_frames
	}
	with open(output_json, 'w') as f:
		json.dump(json_output, f, indent=4)
	
	# **Step 2: Convert JSON to C3D using makeC3d**
	c3d_filename = output_json.replace(".json", ".c3d")
	makeC3d(output_json, c3d_filename)  # Call existing function

	print(f"C3D file saved to {c3d_filename}")
