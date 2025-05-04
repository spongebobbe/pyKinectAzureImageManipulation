import os
import json

from examples.utils.azureKinectFileUtils import makeC3d
import pykinect_azure as pykinect
from pykinect_azure.k4arecord.playback import Playback

from examples.utils.inpaintingUtils import inpaint_blobs_IR
from examples.utils.azureKinectFileUtils import BONE_LIST, extract_skeleton_data, makeC3d 

# Define base input and output directories
BASE_DIR = r"D:\mkv_recordings\VICON KINECT\tmp2"
OUTPUT_DIR = r"D:\mkv_recordings\VICON KINECT\tmp2\processed_data_new2"

# List of subjects to ignore (folders that are not subjects)
IGNORE_FOLDERS = {"DEPTH", "IR", "tmp", "processed_data", "processed_data", "processed_data_simple_fill"}

def process_mkv(video_filename, output_folder, modify_IR=True):
	"""
	Process a single MKV file, extract skeleton data, and save JSON + C3D.
	If modify_IR=True, preprocess the IR image before skeletal tracking.
	"""
	# Create output file names
	base_name = os.path.splitext(os.path.basename(video_filename))[0]  # Extracts "SA" from "SA.mkv"
	json_output = os.path.join(output_folder, f"{base_name}.json")
	c3d_output = os.path.join(output_folder, f"{base_name}.c3d")

	# Initialize the library
	pykinect.initialize_libraries(track_body=True)

	# Start playback
	playback = Playback(video_filename)
	calibration = playback.get_calibration()

	# Start body tracker
	bodyTracker = pykinect.start_body_tracker(calibration=calibration)

	all_frames = []  # Store skeleton data

	frame_count = 0
	while True:
		# Get camera capture
		ret, capture = playback.update()
		if not ret:
			break

		# **Modify IR Image Before Skeletal Tracking (If Enabled)**
		if modify_IR:
			ret_ir, IR_image = capture.get_ir_image()
			if not ret_ir:
				continue 
			modified_ir_image, original_contours, processed_contours, processed_image_clipped, blob_mask = inpaint_blobs_IR(IR_image)

			playback._set_ir_image_to_capture(modified_ir_image)

		# **Run Skeletal Tracking**
		body_frame = bodyTracker.update(capture=capture)
		ret_seg, body_image_color = body_frame.get_segmentation_image()
		if not ret_seg:
			continue

		# **Extract Skeleton Data**
		timestamp_usec = body_frame.get_device_timestamp_usec()
		frame_skeleton_data = extract_skeleton_data(body_frame)
		all_frames.append({
			"bodies": frame_skeleton_data,
			"frame_id": frame_count,
			"num_bodies": len(frame_skeleton_data),
			"timestamp_usec": timestamp_usec
		})
		frame_count += 1

	# **Save Skeleton Data in JSON Format**
	json_output_data = {
		"bone_list": BONE_LIST,
		"frames": all_frames
	}
	with open(json_output, 'w') as f:
		json.dump(json_output_data, f, indent=4)

	# **Convert JSON to C3D**
	makeC3d(json_output, c3d_output)
	print(f"Processed: {video_filename} → {json_output}, {c3d_output}")


if __name__ == "__main__":
	"""
	Process all subjects and MKV files inside the VICON KINECT folder.
	"""
	# Loop through each subject folder
	for subject_folder in os.listdir(BASE_DIR):
		subject_path = os.path.join(BASE_DIR, subject_folder)

		# Skip non-directory files and ignored folders
		if not os.path.isdir(subject_path) or subject_folder in IGNORE_FOLDERS:
			continue

		print(f"Processing subject: {subject_folder}")

		# Locate the VICONKINECT folder inside each subject folder
		viconkinect_path = os.path.join(subject_path, "VICONKINECT")
		if not os.path.isdir(viconkinect_path):
			print(f"Skipping {subject_folder}: No 'VICONKINECT' folder found.")
			continue

		# Create an output directory for the subject
		output_subject_folder = os.path.join(OUTPUT_DIR, subject_folder)
		os.makedirs(output_subject_folder, exist_ok=True)

		# Process each MKV file inside VICONKINECT
		for mkv_file in os.listdir(viconkinect_path):
			if mkv_file.endswith(".mkv"):
				mkv_path = os.path.join(viconkinect_path, mkv_file)
				process_mkv(mkv_path, output_subject_folder, modify_IR=True)
