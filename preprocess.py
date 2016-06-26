import csv
import os

from skimage import io
from skimage import transform
from skimage import util

import numpy as np

DESIRED_SHAPE = (240, 320)
DESIRED_ASPECT_RATIO = float(DESIRED_SHAPE[0]) / float(DESIRED_SHAPE[1])

def preprocessFiles(in_dir, out_file, num_images=-1):
	files = os.listdir(in_dir)
	if num_images > 0:
		files = files[0:num_images]

	# This will store data for all images.
	datas = np.ndarray([len(files), DESIRED_SHAPE[0], DESIRED_SHAPE[1]], dtype=np.float16)

	for i, fname in enumerate(files):
		# Load image and convert to greyscale
		data = io.imread(os.path.join(in_dir, fname), as_grey=True)

		if data.shape[0] > data.shape[1]:
			# Make all images have width be the longest dimension.
			data = transform.rotate(data, 90, resize=True)

		# Scale images so that the shortest dimension matches the corresponding one in the 
		# DESIRED_SHAPE. We do this so that later we can crop a bit of the longer axis and all
		# images the same dimensions without distortion.
		scale_factor = (float(DESIRED_SHAPE[1]) / float(data.shape[1]) 
						if DESIRED_ASPECT_RATIO < float(data.shape[0]) / float(data.shape[1])
						else float(DESIRED_SHAPE[0]) / float(data.shape[0])) 
		data = transform.rescale(data, scale_factor)

		# Crop whatever is left on the long axis
		data = util.crop(data, [(0, data.shape[0] - DESIRED_SHAPE[0]), 
								(0, data.shape[1] - DESIRED_SHAPE[1])])

		datas[i] = data

		print '%s: %s, %0.5f' % (fname, data.shape, float(data.shape[0]) / float(data.shape[1]))

	with open(out_file, 'w') as f:
		f.write(datas.tobytes())

def loadData(in_file, num_images, out_dir=None):
	data = np.fromfile(in_file, dtype=np.float16,
		               count=DESIRED_SHAPE[0] * DESIRED_SHAPE[1] * num_images)
	data = data.reshape(num_images, DESIRED_SHAPE[0], DESIRED_SHAPE[1])
	
	if out_dir:
		for i in range(num_images):
			img_data = data[i]
			io.imsave(os.path.join(out_dir, '%d.jpg' % i), img_data)

	return data

def preprocessLabels(in_file, out_file):
	labels = []
	with open(in_file) as f:
		reader = csv.reader(f)

		# The training CSV containts 'begnin'/'malignant' while the test
		# CSV contains '0.0/1.0'. Go figure!
		labels = [0.0 if (row[1] == 'benign' or row[1] == '0.0') else 1.0 
		          for row in reader]

		data = np.asarray(labels, dtype=np.float16)

		with open(out_file, 'w') as f:
			f.write(data.tobytes())

def loadLabels(in_file, num):
	labels = np.fromfile(in_file, dtype=np.float16, count=num)
	print(labels)
	

if __name__ == '__main__':
	scratch_dir = '/Users/smike/Downloads/ISBI2016_ISIC-scratch'
	training_data_input_dir = '/Users/smike/Downloads/ISBI2016_ISIC_Part3_Training_Data'
	training_labels_input = '/Users/smike/Downloads/ISBI2016_ISIC_Part3_Training_GroundTruth.csv'
	training_data_bin_file = '/Users/smike/Downloads/ISBI2016_ISIC_Part3_Training_Data-prepared.bin'
	training_label_bin_file = '/Users/smike/Downloads/ISBI2016_ISIC_Part3_Training_GroundTruth.bin'

	test_data_input_dir = '/Users/smike/Downloads/ISBI2016_ISIC_Part3_Test_Data'
	test_labels_input = '/Users/smike/Downloads/ISBI2016_ISIC_Part3_Test_GroundTruth.csv'
	test_data_bin_file = '/Users/smike/Downloads/ISBI2016_ISIC_Part3_Test_Data-prepared.bin'
	test_label_bin_file = '/Users/smike/Downloads/ISBI2016_ISIC_Part3_Test_GroundTruth.bin'

	preprocessFiles(training_data_input_dir, training_data_bin_file)  # , num_images=10)
	preprocessLabels(training_labels_input, training_label_bin_file)
	loadData(training_data_bin_file, 10, out_dir=scratch_dir)
	loadLabels(training_label_bin_file, 10)

	preprocessFiles(test_data_input_dir, test_data_bin_file)  # , num_images=10)
	preprocessLabels(test_labels_input, test_label_bin_file)
	loadData(test_data_bin_file, 10, out_dir=scratch_dir)
	loadLabels(test_label_bin_file, 10)