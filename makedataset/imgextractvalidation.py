import numpy as np
import os
from PIL import Image, ImageOps

def _load_mlt():
	training_y = []
	training_x = []
	testing_x = []
	testing_y = []
	validation_x = []
	validation_y = []

	cur_dir = os.path.dirname(os.path.abspath(__file__))
	folders = ["auto", "car", "bicycle", "motorcycle", "person"]
	obj_id = [0, 1, 2, 3, 4]
	size = (36, 36)

	data_dir = os.path.join(cur_dir, "train")
	for i in range(0,5):
		folder = folders[i]
		obj_dir = os.path.join(data_dir, folder)
		files = os.listdir(obj_dir)
		for file in files:
			col = Image.open(obj_dir+"/"+file)
			gray = col.convert('L')
			gray.thumbnail(size, Image.ANTIALIAS)
			background = Image.new('L', size)
			background.paste( gray, ((size[0] - gray.size[0]) / 2, (size[1] - gray.size[1]) / 2))
			
			bw = np.asarray(background)
			training_x.append(bw)
			training_y.append(obj_id[i])

	training_x = np.asarray(training_x)
	training_y = np.asarray(training_y)
	training_x = training_x.reshape((training_x.shape[0], 1, size[0], size[1]))
	print training_x.shape

	###############test-set
	data_dir = os.path.join(cur_dir, "test")
	for i in range(0,5):
		folder = folders[i]
		obj_dir = os.path.join(data_dir, folder)
		files = os.listdir(obj_dir)
		for file in files:
			col = Image.open(obj_dir+"/"+file)
			gray = col.convert('L')
			gray.thumbnail(size, Image.ANTIALIAS)
			background = Image.new('L', size)
			background.paste( gray, ((size[0] - gray.size[0]) / 2, (size[1] - gray.size[1]) / 2))
			
			bw = np.asarray(background)
			testing_x.append(bw)
			testing_y.append(obj_id[i])

	testing_x = np.asarray(testing_x)
	testing_y = np.asarray(testing_y)
	testing_x = testing_x.reshape((testing_x.shape[0], 1, size[0], size[1]))

	###############validation-set
	data_dir = os.path.join(cur_dir, "validation")
	for i in range(0,5):
		folder = folders[i]
		obj_dir = os.path.join(data_dir, folder)
		files = os.listdir(obj_dir)
		for file in files:
			col = Image.open(obj_dir+"/"+file)
			gray = col.convert('L')
			gray.thumbnail(size, Image.ANTIALIAS)
			background = Image.new('L', size)
			background.paste( gray, ((size[0] - gray.size[0]) / 2, (size[1] - gray.size[1]) / 2))
			
			bw = np.asarray(background)
			validation_x.append(bw)
			validation_y.append(obj_id[i])

	validation_x = np.asarray(validation_x)
	validation_y = np.asarray(validation_y)
	validation_x = validation_x.reshape((validation_x.shape[0], 1, size[0], size[1]))
	
	return training_x, training_y, testing_x, testing_y, validation_x, validation_y

training_x, training_y, testing_x, testing_y, validation_x, validation_y = _load_mlt()


