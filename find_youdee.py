import tensorflow.keras
from PIL import Image, ImageOps
from matplotlib import cm
import numpy as np
import sys
import os.path
from os import listdir
from collections import defaultdict
import cv2
from dataclasses import dataclass

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# standardize image/video-frame size to 224x224 used during training
size = (224, 224)

# For an image, what hould the minimum confidence be for a frame containing YoUDee
image_percent_threshold = 0.85

# For a video, what should the minimum confidence be for a frame containing YoUDee
video_percent_threshold = 0.80

# For a video, how many consecutive frames of YoUDee for it to be significant YoUDee in video
video_consecutive_threshold = 7

# OpenCV2's support and sometimes supported image formats
# See: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#imread
supported_image_formats = ['bmp', 'dib', 'pbm', 'pgm', 'ppm', 'sr', 'ras']
suspect_image_formats = ['jpeg', 'jpg', 'jpe', 'jp2', 'png', 'tiff', 'tif']

supported_video_formats = ['mp4', 'mov', 'gif']

def load_image(path):
	image_array = None
	# resize the image to a 224x224 with the same strategy as in TM2:
	# resizing the image to be at least 224x224 and then cropping from the center
	image = Image.open(path)
	image = ImageOps.fit(image, size, Image.ANTIALIAS)
	#image = image.resize(size)
	# turn the image into a numpy array
	image_array = np.asarray(image)
	# display the resized image
	#image.show()
	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
	return normalized_image_array

def process_image(path):
	# Create the array of the right shape to feed into the keras model
	# The 'length' or number of images you can put into the array is
	# determined by the first position in the shape tuple, in this case 1.
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	# Load the image into the array
	data[0] = load_image(path)
	return data

def load_video(path):
	vidcap = cv2.VideoCapture(path)
	success,image = vidcap.read()
	frames = []
	load_one_frame = True
	while success:
		# One of the fun quirks of OpenCV2 that's not adequately stated anywhere
		# is that cv2 images are BGR, while pillow images are RGB (different
		# order of color values.) Must be converted before passing to keras
		# model prediction.
		image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
		image = Image.fromarray(image, 'RGB')
		image = ImageOps.fit(image, size, Image.ANTIALIAS)
		#image = image.resize(size)
		#if load_one_frame:
		#	image.show()
		#	load_one_frame = False
		image_array = np.asarray(image)
		frames.append((image_array.astype(np.float32) / 127.0) - 1)
		success,image = vidcap.read()
	return frames

@dataclass
class ProcessedVideo:
	video_ndarray: np.ndarray
	frame_count: int

def process_video(path):
	frames = load_video(path)
	frame_count = len(frames)
	video_ndarray = np.ndarray(shape=(frame_count, 224, 224, 3), dtype=np.float32)
	for i in range(0, frame_count):
		video_ndarray[i] = frames[i]
	return ProcessedVideo(video_ndarray, frame_count)

def get_labels(path):
	"""
	Get labels as a standard dictionary, key with first column, value with
	second column of a labels.txt file.

	Example:
	```
	0 YoUDee
	1 Neutral
	```
	Becomes
	```
	{0: 'YoUDee', 1: 'Neutral'}
	"""
	lines = False
	labels = {}
	with open('labels.txt') as f:
		lines = f.readlines()
	for l in lines:
		l = l.rstrip().split(' ', 1)
		labels[int(l[0])] = l[1]
	return labels

def process_labels(path):
	# load labels
	labels = get_labels(path)
	original_label_avg = {}
	for label in labels:
		original_label_avg[label] = 0.0
	return original_label_avg

def get_arguments():
	arguments = defaultdict(list)
	for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in sys.argv[1:])):
		arguments[k].append(v)

	return dict(arguments)

def check_arguments(arguments):
	if len(arguments) > 1:
		sys.exit('Too many arguments for "image"')
	if os.path.isfile(arguments[0]) is False:
		sys.exit('File does not exist "' + arguments['image'][0] + '"')

@dataclass
class Job:
	type: str
	data: np.ndarray
	filename: str
	frame_count: int
	result: list
	has_youdee: bool

arguments = get_arguments()
jobs = []

# load labels
labels = get_labels('labels.txt')
zeroed_labels = process_labels('labels.txt')

if 'image' in arguments:
	check_arguments(arguments['image'])

	jobs.append(Job('image', process_image(arguments['image'][0]), \
	 arguments['image'][0], 1, None, False))

elif 'video' in arguments:
	check_arguments(arguments['video'])

	p_video = process_video(arguments['video'][0])
	jobs.append(Job('video', p_video.video_ndarray, arguments['video'][0], \
	 p_video.frame_count, None, False))

elif 'dir' in arguments:
	print('Reading all files in directory...')
	files = listdir(arguments['dir'][0])

	for f in files:
		print('Reading file "' + f + '"...')
		filename, file_extension = os.path.splitext(f)
		file_extension = file_extension.lower().replace('.', '', 1)
		filepath = arguments['dir'][0] + f

		if file_extension in supported_video_formats:
			p_video = process_video(filepath)
			print('Found file ' + f + ' with ' + str(p_video.frame_count) + ' frames')
			jobs.append(Job('video', p_video.video_ndarray, filepath, \
			 p_video.frame_count, None, False))
		elif file_extension in supported_image_formats or file_extension in suspect_image_formats:
			if file_extension in suspect_image_formats:
				print('"' + f + '" file type of "' + file_extension + '" may not be a supported image format on your OS')
			jobs.append(Job('image', process_image(filepath), filepath, 1, None, False))
		else:
			print('File format "' + file_extension + '" not supported for file "' + f + '"')
	print('Finished reading all files in directory')
	if len(files) == 0:
		sys.exit('No files in directory')
	if len(jobs) == 0:
		sys.exit('Found no readable files in directory')
elif 'help' in arguments:
	print('--image=[FILE_PATH]  ... predict one image')
	print('--video=[FILE_PATH]  ... predict one video')
	print('--dir=[DIR_PATH]     ... predict all valid image and video files in a directory')
	print('--output=[FILE_PATH] ... where to save csv of prediction results')
	sys.exit()
else:
	sys.exit('No arguments specified')

# run the inference
model = tensorflow.keras.models.load_model('keras_model.h5', {'YoUDee': 'YoUDee', 'Neutral': 'Neutral'})

for j in jobs:
	print('Doing job ' + j.filename + ' type of ' + j.type + ' with ' + str(j.frame_count) + ' frames')
	j.result = model.predict(j.data)

sustained_frames = 0

for j in jobs:
	print(j.filename)
	sustained_frames = 0
	labels_avg = zeroed_labels.copy()
	for i in range(0, j.frame_count):
		if j.type == 'video':
			if j.result[i][0] > video_percent_threshold:
				if sustained_frames > video_consecutive_threshold:
					j.has_youdee = True
				sustained_frames += 1
			else:
				sustained_frames = 0
		for label in labels:
			if j.type == 'image':
				print(str(labels[label]) + ": " + \
				str(round(j.result[0][label] * 100, 3)) + "%")
			elif j.type == 'video':
				labels_avg[label] += j.result[i][label]
		if j.type == 'image':
			if j.result[0][0] > image_percent_threshold:
				j.has_youdee = True
			print('Has YouDee? ' + str(j.has_youdee))
	if j.type == 'video':
		for label in labels:
			print(labels[label] + ": " + \
			str(round((labels_avg[label] / float(j.frame_count)) * 100, 3)) + "%")
		print('Has YouDee? ' + str(j.has_youdee))

if 'output' in arguments:
	with open(arguments['output'][0], 'a') as output:
		output.write('type,filename,frame_count,youdee_result,neutral_result,has_youdee\n')
		for j in jobs:
			output.write(j.type + ',' + j.filename + ',' + str(j.frame_count) + \
			str(j.result[0][0]) + ',' + str(j.result[0][1]) + ',' + str(j.has_youdee) + \
			'\n')
	print('Saved output in "' + arguments['output'][0])
