import os
import cv2
import numpy as np
import json

from examples.utils.azureKinectFileUtils import makeC3d
import pykinect_azure as pykinect
from pykinect_azure.k4arecord.playback import Playback

from examples.utils.inpaintingUtils import inpaint_blobs_IR, clip_IR, visualize_ir_before_after
from examples.utils.azureKinectFileUtils import BONE_LIST, extract_skeleton_data, makeC3d 

if __name__ == "__main__":
	plot = True  # Set to True to enable plotting
	modify_depth = False  # Set to True to modify depth image
	modify_IR = True
	modify_RGB = False  # not implemented
	# Ensure we never try to do both at once
	assert not (modify_depth and modify_IR), "modify_depth and modify_IR cannot both be True"
	video_filename = 'output.mkv'
	output_json = 'skeleton_data.json' 

	pykinect.initialize_libraries(track_body=True)
	playback = Playback(video_filename)
	calibration = playback.get_calibration()
	bodyTracker = pykinect.start_body_tracker(calibration=calibration)
	all_frames = []  # Store skeleton data for all frames

	frame_count = 0
	# Process a single MKV file, extract skeleton data, and save JSON + C3D.
	while True:
		# Get camera capture
		ret, capture = playback.update()
		if not ret:
			break

		# Modify the depth image BEFORE skeletal tracking (if modify_depth is set to True)
		if modify_depth and not(modify_IR):
			ret_depth, depth_image = capture.get_depth_image()
			if ret_depth and depth_image is not None:
				#modified_depth_image = np.zeros_like(depth_image, dtype=np.uint16)
				modified_depth_image = np.ascontiguousarray(np.fliplr(depth_image))
				playback._set_depth_image_to_capture(modified_depth_image)
			
		if modify_RGB:
			pass # not implemented
						
		ret_ir, IR_image = capture.get_ir_image()
		ir_original = clip_IR(IR_image, uint_8=True)
		cv2.imshow('Infrared Image (clipped) before inpainting',ir_original)
		if modify_IR and not(modify_depth):
			ret_ir, IR_image = capture.get_ir_image()
			if ret_ir and IR_image is not None:
				modified_ir_image, original_contours, processed_contours, processed_image_clipped, blob_mask = inpaint_blobs_IR(IR_image)
				visualize_ir_before_after(original_contours, processed_contours, processed_image_clipped)
			
				#to demonstrate that if ir is flipped and depth is not flipped, the skeleton will be flipped
				#modified_ir_image = np.zeros_like(IR_image, dtype=np.uint16)
				#modified_ir_image = np.ascontiguousarray(np.fliplr(IR_image))
				#playback._set_ir_image_to_capture(modified_ir_image)

				ret_ir, IR_image = capture.get_ir_image()
				ir_inpainted = clip_IR(IR_image, uint_8=True)
				cv2.imshow('Infrared Image (clipped) after inpainting', ir_inpainted)
			
		'''
		if (modify_depth and modify_IR): 
			ret_depth, depth_image = capture.get_depth_image()
			if ret_depth and depth_image is not None:
				modified_depth_image = inpaint_blobs_depth(depth_image,blob_mask)
				playback._set_depth_image_to_capture(modified_depth_image)
		'''		
		
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

		if plot:
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

			# Display the image
			cv2.imshow('Depth with Skeleton', combined_image) 
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		if frame_count ==372 and modify_IR: #save images for a specific frame
			output_dir = "saved_images"
			cv2.imwrite(os.path.join(output_dir, "infrared_original.png"), ir_original)
			cv2.imwrite(os.path.join(output_dir, "infrared_inpainted.png"), ir_inpainted)
			cv2.imwrite(os.path.join(output_dir, "depth_with_skeleton.png"), combined_image)
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
