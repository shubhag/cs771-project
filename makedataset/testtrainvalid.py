import os
import numpy
from shutil import copyfile
import shutil

def check_folder(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(cur_dir, "total")
train_dir = os.path.join(cur_dir, "train")
check_folder(train_dir)
test_dir = os.path.join(cur_dir, "test")
check_folder(test_dir)
valid_dir = os.path.join(cur_dir, "validation")
check_folder(valid_dir)
folders = ["auto", "car", "bicycle", "motorcycle", "person","rickshaw"]
for folder in folders:
	d_dir = os.path.join(dataset_dir, folder)
	tra_dir = os.path.join(train_dir, folder)
	tes_dir = os.path.join(test_dir, folder)
	val_dir = os.path.join(valid_dir, folder)
	check_folder(tra_dir)
	check_folder(tes_dir)
	check_folder(val_dir)
	files = os.listdir(d_dir)
	for i in range(0, len(files)):
		if i%10 == 0:
			copyfile(d_dir+"/"+files[i], tes_dir+"/"+files[i])
		elif i%10 == 1:
			copyfile(d_dir+"/"+files[i], val_dir+"/"+files[i])
		else:
			copyfile(d_dir+"/"+files[i], tra_dir+"/"+files[i])

