import json
import c3d
import numpy as np

# Bone connections based on Azure Kinect SDK structure
BONE_LIST = [
	["SPINE_CHEST", "SPINE_NAVEL"], ["SPINE_NAVEL", "PELVIS"],
	["SPINE_CHEST", "NECK"], ["NECK", "HEAD"], ["HEAD", "NOSE"],
	["SPINE_CHEST", "CLAVICLE_LEFT"], ["CLAVICLE_LEFT", "SHOULDER_LEFT"],
	["SHOULDER_LEFT", "ELBOW_LEFT"], ["ELBOW_LEFT", "WRIST_LEFT"],
	["WRIST_LEFT", "HAND_LEFT"], ["HAND_LEFT", "HANDTIP_LEFT"],
	["WRIST_LEFT", "THUMB_LEFT"], ["PELVIS", "HIP_LEFT"],
	["HIP_LEFT", "KNEE_LEFT"], ["KNEE_LEFT", "ANKLE_LEFT"],
	["ANKLE_LEFT", "FOOT_LEFT"], ["NOSE", "EYE_LEFT"],
	["EYE_LEFT", "EAR_LEFT"], ["SPINE_CHEST", "CLAVICLE_RIGHT"],
	["CLAVICLE_RIGHT", "SHOULDER_RIGHT"], ["SHOULDER_RIGHT", "ELBOW_RIGHT"],
	["ELBOW_RIGHT", "WRIST_RIGHT"], ["WRIST_RIGHT", "HAND_RIGHT"],
	["HAND_RIGHT", "HANDTIP_RIGHT"], ["WRIST_RIGHT", "THUMB_RIGHT"],
	["PELVIS", "HIP_RIGHT"], ["HIP_RIGHT", "KNEE_RIGHT"],
	["KNEE_RIGHT", "ANKLE_RIGHT"], ["ANKLE_RIGHT", "FOOT_RIGHT"],
	["NOSE", "EYE_RIGHT"], ["EYE_RIGHT", "EAR_RIGHT"]
]


def extract_skeleton_data(body_frame):
	"""
	Extracts skeleton joint data from the body frame.
	Returns a list of dictionaries for each detected body.
	"""
	bodies = body_frame.get_bodies()
	skeleton_data = []

	for body in bodies:
		body_id = body.id
		joints = body.joints

		joint_orientations = []
		joint_positions = []
		joint_confidences = []

		for joint_idx, joint in enumerate(joints):
			joint_orientations.append([joint.orientation.w, joint.orientation.x, joint.orientation.y, joint.orientation.z])
			joint_positions.append([joint.position.x, joint.position.y, joint.position.z])
			joint_confidences.append(joint.confidence_level)

		skeleton_data.append({
			"body_id": body_id,
			"joint_orientations": joint_orientations,
			"joint_positions": joint_positions,
			"joint_confidences": joint_confidences
		})
	
	return skeleton_data
class KinectAzureFile(object):
	def __init__(self, filename):
		f = open(filename, "r")
		self._data = json.loads(f.read())

	def getJointNames(self):
		return self._data.joint_names

	def getNrFrames(self):
		return len(self._data['frames'])

	def getNrPoints(self):
		return 32
		#return len(self._data['frames'][0]['bodies'][0]['joint_positions'])

	def getJointPositions(self,framenr):
		return self._data['frames'][framenr]['bodies'][0]['joint_positions']

	def getLabels(self):
		return [
		"PELVIS", "SPINE_NAVEL", "SPINE_CHEST", "NECK", 
		"CLAVICLE_LEFT", "SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT", 
		"HAND_LEFT", "HANDTIP_LEFT", "THUMB_LEFT", 
		"CLAVICLE_RIGHT", "SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT", 
		"HAND_RIGHT", "HANDTIP_RIGHT", "THUMB_RIGHT", 
		"HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT", "FOOT_LEFT", 
		"HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT", "FOOT_RIGHT", 
		"HEAD", "NOSE", "EYE_LEFT", "EAR_LEFT", "EYE_RIGHT", "EAR_RIGHT"
		]


def makeC3d(filein, fileout):
	test1 = KinectAzureFile(filein)

	w = c3d.Writer(point_rate=30)  # Create a C3D writer with 30 FPS

	# Set the joint labels explicitly
	w.set_point_labels(test1.getLabels())  

	nrPoints = test1.getNrPoints()
	
	for i in range(test1.getNrFrames()):
		points = np.empty((nrPoints, 5))
		
		for p in range(nrPoints):
			try:
				data = test1.getJointPositions(i)[p]
			except:
				data = [0, 0, 0]
			
			points[p][0] = data[0]
			points[p][1] = data[1]
			points[p][2] = data[2]
			points[p][3] = 0  # Residual
			points[p][4] = 1  # Camera visibility (default to 1)

		analog = np.empty((0, 0))  # No analog data
		frame1 = (points, analog)
		w.add_frames([frame1])

	# Save the C3D file
	with open(fileout, 'wb') as handle:
		w.write(handle)  # No 'labels' argument

	print(f"C3D file saved to {fileout}")
