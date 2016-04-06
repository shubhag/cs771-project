import os
import numpy
from shutil import copyfile
import shutil

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(cur_dir, "total")
train_dir = os.path.join(cur_dir, "train")
test_dir = os.path.join(cur_dir, "test")
valid_dir = os.path.join(cur_dir, "validation")
folders = ["auto", "car", "bicycle", "motorcycle", "person","rickshaw"]
for folder in folders:
	d_dir = os.path.join(dataset_dir, folder)
	tra_dir = os.path.join(train_dir, folder)
	tes_dir = os.path.join(test_dir, folder)
	val_dir = os.path.join(valid_dir, folder)
	files = os.listdir(d_dir)
	for i in range(0, len(files)):
		if i%9 == 0:
			copyfile(d_dir+"/"+files[i], tes_dir+"/"+files[i])
		else:
			copyfile(d_dir+"/"+files[i], tra_dir+"/"+files[i])

