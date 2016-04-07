import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from sklearn.svm import LinearSVC
from PIL import Image, ImageOps

def getData(folder):
	FV = []
	FL = []
	cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	cur_dir = os.path.join(cur_dir, "makedataset")
	folders = ["auto", "car", "bicycle", "motorcycle", "person"]
	obj_id = [0, 1, 2, 3, 4]
	size = (36, 36)

	data_dir = os.path.join(cur_dir, folder)
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
			FV.append(bw)
			FL.append(obj_id[i])

	FV = np.asarray(FV)
	FL = np.asarray(FL)
	# FV = FV.reshape((FV.shape[0], 1, size[0], size[1]))
	print FV.shape
	return FV,FL

def constructHoG(data, orientations = 8, cell_size = (4,4), block_size = (2,2) ):
	res = []
	for img in data:
		FV = hog(img, orientations, cell_size, block_size)
		res.append(FV)
	return np.asarray(res)

if __name__ == '__main__':
	trainData, trainLabel = getData("train")
	validationData, validationLabel = getData("validation")
	testData, testLabel = getData("test")
	
	print "Data Loaded..."

	svm_accuracy = np.zeros(shape = (16,16)) #from 3 to 18

	best_orientation = -1
	best_cell = -1
	best_accuracy = -1

	for i in range(16):
		o = i+3
		for j in range(16):
			c = j+3
			trainHoG = constructHoG(trainData, orientations = o, cell_size = (c,c))
			validationHoG = constructHoG(validationData, orientations = o, cell_size = (c,c))
			print "Constructed HoG for orientation =",o,"and cell_size =",(c,c)
			SVM = LinearSVC(loss = 'hinge')
			SVM.fit(trainHoG,trainLabel)
			svm_accuracy[i,j] = SVM.score(validationHoG, validationLabel)*100.0
			print svm_accuracy[i,j]
			if svm_accuracy[i,j] > best_accuracy:
				best_accuracy = svm_accuracy[i,j]
				best_orientation = o
				best_cell = c 
	np.savetxt("svm_training_stats.txt",svm_accuracy,fmt = '%10.5f')

	print "Finding accuracy for best found parameters:"
	print "Best cell size:",best_cell
	print "Best orientations:", best_orientation

	trainHoG = constructHoG(trainData, orientations = best_orientation, cell_size = (best_cell,best_cell))
	testHoG = constructHoG(testData, orientations = best_orientation, cell_size = (best_cell,best_cell))
	SVM = LinearSVC(loss = 'hinge')
	SVM.fit(trainHoG,trainLabel)
	print "Accuracy: ",SVM.score(testHoG, testLabel)*100.0
