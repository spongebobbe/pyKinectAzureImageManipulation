from pykinect_azure.k4arecord import _k4arecord
from pykinect_azure.k4arecord.datablock import Datablock
from pykinect_azure.k4arecord.record_configuration import RecordConfiguration

from pykinect_azure.k4a import _k4a
from pykinect_azure.k4a.capture import Capture
from pykinect_azure.k4a.calibration import Calibration
from pykinect_azure.k4a.imu_sample import ImuSample
import numpy as np
import ctypes


class Playback:

	def __init__(self, filepath):

		self._handle = _k4arecord.k4a_playback_t()
		self._capture = None
		self._datablock = None

		self.open(filepath)
		self.calibration = self.get_calibration()

	def __del__(self):
		self.close()

	def open(self, filepath):

		_k4arecord.VERIFY(_k4arecord.k4a_playback_open(filepath.encode('utf-8'),self._handle),"Failed to open recording!")

	def update(self):
		return self.get_next_capture()

	def is_valid(self):
		return self._handle != None

	def is_capture_initialized(self):
		return self._capture

	def is_datablock_initialized(self):
		return self._datablock

	def close(self):
		if self.is_valid():
			_k4arecord.k4a_playback_close(self._handle)
			self._handle = None

	def get_calibration(self):
		calibration_handle = _k4arecord.k4a_calibration_t()
		if self.is_valid():
			_k4arecord.VERIFY(_k4arecord.k4a_playback_get_calibration(self._handle, calibration_handle),"Failed to read device calibration from recording!")

		return Calibration(calibration_handle)

	def get_record_configuration(self):
		config = _k4arecord.k4a_record_configuration_t()

		if self.is_valid():		
			_k4arecord.VERIFY(_k4arecord.k4a_playback_get_record_configuration(self._handle, config),"Failed to read record configuration!")

		return RecordConfiguration(config)

	def get_next_capture(self):
		capture_handle = _k4a.k4a_capture_t()

		if self.is_capture_initialized():
			self._capture.release_handle()
			self._capture._handle = capture_handle
		else:
			self._capture = Capture(capture_handle, self.calibration)

		ret = _k4arecord.k4a_playback_get_next_capture(self._handle, capture_handle) != _k4arecord.K4A_STREAM_RESULT_EOF


		return ret, self._capture

	def get_previous_capture(self):
		capture_handle = _k4a.k4a_capture_t()

		if self.is_capture_initialized():
			self._capture.release_handle()
			self._capture._handle = capture_handle
		else:
			self._capture = Capture(capture_handle, self.calibration)

		ret = _k4arecord.k4a_playback_get_previous_capture(self._handle, capture_handle) != _k4arecord.K4A_STREAM_RESULT_EOF

		return ret, self._capture
	
	def get_next_imu_sample(self):
		imu_sample_struct = _k4a.k4a_imu_sample_t()
		_k4a.VERIFY(_k4arecord.k4a_playback_get_next_imu_sample(self._handle, imu_sample_struct),"Get next imu sample failed!")
			
		# Convert the structure into a dictionary
		_imu_sample = ImuSample(imu_sample_struct)

		return _imu_sample

	def get_previous_imu_sample(self):
		imu_sample_struct = _k4a.k4a_imu_sample_t()
		_k4a.VERIFY(_k4arecord.k4a_playback_get_previous_imu_sample(self._handle, imu_sample_struct),"Get previous imu sample failed!")
			
		# Convert the structure into a dictionary
		_imu_sample = ImuSample(imu_sample_struct)

		return _imu_sample

	def seek_timestamp(self, offset = 0, origin = _k4arecord.K4A_PLAYBACK_SEEK_BEGIN):
		_k4a.VERIFY(_k4arecord.k4a_playback_seek_timestamp(self._handle, offset, origin),"Seek recording failed!")
			
	def get_recording_length(self):
		return int(_k4arecord.k4a_playback_get_recording_length_usec(self._handle))

	def set_color_conversion(self, format = _k4a.K4A_IMAGE_FORMAT_DEPTH16):
		_k4a.VERIFY(_k4arecord.k4a_playback_set_color_conversion(self._handle, format),"Seek color conversio failed!")

	def get_next_data_block(self, track):
		block_handle = _k4arecord.k4a_playback_data_block_t()
		_k4a.VERIFY(_k4arecord.k4a_playback_get_next_data_block(self._handle, track, block_handle),"Get next data block failed!")
			
		if self.is_datablock_initialized():
			self._datablock._handle = block_handle
		else :
			self._datablock = Datablock(block_handle)

		return self._datablock

	def get_previous_data_block(self, track):
		block_handle = _k4arecord.k4a_playback_data_block_t()
		_k4a.VERIFY(_k4arecord.k4a_playback_get_previous_data_block(self._handle, track, block_handle),"Get previous data block failed!")
			
		if self.is_datablock_initialized():
			self._datablock._handle = block_handle
		else :
			self._datablock = Datablock(block_handle)

		return self._datablock

	def get_current_capture(self):
		"""Returns the current capture without advancing to the next frame."""
		if self.is_capture_initialized():
			return self._capture  # Return the existing capture
		else:
			return None  # No capture available

	def _set_depth_image_to_capture(self, depth_image_np):
		"""Private method to inject a modified depth image into the Capture object."""
		height, width = depth_image_np.shape
		depth_image_size = width * height * 2  # 2 bytes per pixel for 16-bit depth image

		# Create a k4a_image_t from numpy array
		depth_image_handle = _k4a.k4a_image_t()
		_k4a.VERIFY(
			_k4a.k4a_image_create(_k4a.K4A_IMAGE_FORMAT_DEPTH16, width, height, width * 2, ctypes.byref(depth_image_handle)),
			"Failed to create depth image!"
		)

		# Copy data from numpy array to k4a_image buffer
		depth_image_ptr = _k4a.k4a_image_get_buffer(depth_image_handle)
		ctypes.memmove(depth_image_ptr, depth_image_np.ctypes.data, depth_image_size)

		 # **Increase reference count before modifying**	
		_k4a.k4a_capture_reference(self._capture._handle)

		# Attach the new depth image to the capture
		_k4a.k4a_capture_set_depth_image(self._capture._handle, depth_image_handle)

		_k4a.k4a_image_release(depth_image_handle)


	def _set_color_image_to_capture(self, color_image_np):
		"""Private method to inject a modified color image into the Capture object."""
		height, width, channels = color_image_np.shape
		color_image_size = width * height * 4  # 8-bit per channel, so no multiplication by 2

		# Create a k4a_image_t from numpy array
		color_image_handle = _k4a.k4a_image_t()
		_k4a.VERIFY(
			_k4a.k4a_image_create(_k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32, width, height, width * 4, ctypes.byref(color_image_handle)),
			"Failed to create color image!"
		)

		# Copy data from numpy array to k4a_image buffer
		color_image_ptr = _k4a.k4a_image_get_buffer(color_image_handle)
		ctypes.memmove(color_image_ptr, color_image_np.ctypes.data, color_image_size)

		# **Increase reference count before modifying**	
		_k4a.k4a_capture_reference(self._capture._handle)

		
		# Attach the new color image to the capture
		_k4a.k4a_capture_set_color_image(self._capture._handle, color_image_handle)

		# Release the handle
		_k4a.k4a_image_release(color_image_handle)


	def _set_ir_image_to_capture(self, ir_image_np):
		"""Private method to inject a modified IR image into the Capture object."""
		height, width = ir_image_np.shape
		ir_image_size = width * height * 2  # 16-bit grayscale IR image

		# Create a k4a_image_t from numpy array
		ir_image_handle = _k4a.k4a_image_t()
		_k4a.VERIFY(
			_k4a.k4a_image_create(_k4a.K4A_IMAGE_FORMAT_IR16, width, height, width * 2, ctypes.byref(ir_image_handle)),
			"Failed to create IR image!"
		)

		# Copy data from numpy array to k4a_image buffer
		ir_image_ptr = _k4a.k4a_image_get_buffer(ir_image_handle)
		ctypes.memmove(ir_image_ptr, ir_image_np.ctypes.data, ir_image_size)

		 # **Increase reference count before modifying**	
		_k4a.k4a_capture_reference(self._capture._handle)

		# Attach the new IR image to the capture
		_k4a.k4a_capture_set_ir_image(self._capture._handle, ir_image_handle)

		# Release the handle
		_k4a.k4a_image_release(ir_image_handle)