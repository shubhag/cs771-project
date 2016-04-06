import numpy as np
import os
from PIL import Image, ImageOps

def _print_info(name, data_set):
    x, y = data_set
    print("""{}
  X::
    shape:{}
    min:{} mean:{:5.2f} max:{:5.2f}
  Y::
    shape:{}
    min:{} mean:{:5.2f} max:{}
    """.format(name,
               x.shape, x.min(), x.mean(), x.max(),
               y.shape, y.min(), y.mean(), y.max(),))


def mlt():
	training_y = []
	training_x = []

	testing_x = []
	testing_y = []
	
	validation_x = []
	validation_y = []

	cur_dir = os.path.dirname("data/")
	folders = ["auto", "car", "bicycle", "motorcycle", "person"]
	obj_id = [0, 1, 2, 3, 4]
	size = (28, 28)

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
			training_y.append(obj_id)

	training_x = np.asarray(training_x)
	training_y = np.asarray(training_y)
	training_x = training_x.reshape((training_x.shape[0], 1, 28, 28))
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
			testing_y.append(obj_id)

	testing_x = np.asarray(testing_x)
	testing_y = np.asarray(testing_y)
	testing_x = testing_x.reshape((testing_x.shape[0], 1, 28, 28))

	###############valid-set
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
			validation_y.append(obj_id)

	validation_x = np.asarray(validation_x)
	validation_y = np.asarray(validation_y)
	validation_x = validation_x.reshape((validation_x.shape[0], 1, 28, 28))
	
	return training_x, training_y, testing_x, testing_y, validation_x, validation_y

training_x, training_y, testing_x, testing_y, validation_x, validation_x = mlt()

if __name__ == '__main__':
    _print_info("Training Data Set:", (training_x, training_y))
    _print_info("Test Data Set:", (testing_x, testing_y))
