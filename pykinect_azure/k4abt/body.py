import numpy as np

from pykinect_azure.k4abt._k4abtTypes import K4ABT_JOINT_COUNT
from pykinect_azure.k4abt.joint import Joint

class Body:
	def __init__(self, skeleton_handle):

		if skeleton_handle:
			self._handle = skeleton_handle
			self.id = getattr(skeleton_handle, 'id', None)  # Attempt to retrieve the body ID
			self.skeleton = getattr(skeleton_handle, 'skeleton', None)  # Expose the skeleton
			self.initialize()

	def json(self):

		return self._handle.__iter__()

	def numpy(self):
		return np.array([joint.numpy() for joint in self.joints])

	def __del__(self):
		self.destroy()

	def is_valid(self):
		return self._handle

	def handle(self):
		return self._handle

	def destroy(self):
		if self.is_valid():
			self._handle = None

	def initialize(self):
		joints = np.ndarray((K4ABT_JOINT_COUNT,),dtype=np.object_)
		if self.skeleton:  # Ensure skeleton exists before accessing joints
			for i in range(K4ABT_JOINT_COUNT):
				joints[i] = Joint(self.skeleton.joints[i], i)
		self.joints = joints

	def __str__(self):

		"""Print the current settings and a short explanation"""
		message = '\nBody:\n' +''.join(f'{joint.__str__()}' for joint in self.joints)

		return message


