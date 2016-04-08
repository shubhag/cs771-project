import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image, ImageOps

def getData(folder):
	FV = []
	FL = []
	cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	cur_dir = os.path.join(cur_dir, "makedataset")
	folders = ["auto", "car", "bicycle", "motorcycle", "person", "rickshaw"]
	obj_id = [0, 1, 2, 3, 4, 5]
	size = (36, 36)

	data_dir = os.path.join(cur_dir, folder)
	for i in range(0,6):
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

def constructHoG(data, orientations = 3, cell_size = (7,7), block_size = (2,2) ):
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

	trainHoG = constructHoG(trainData)
	validationHoG = constructHoG(validationData)
	testHoG = constructHoG(testData)
	
	ada_boost_accuracy = np.zeros(shape = (10,10))
	print "Adaboost Classifiction"
	best_depth = 12
	best_trees = 140
	'''
	best_accuracy = -1

	for i in range(10):
		max_depth = (i+1)*4
		for j in range(10):
			trees = (j+1)*20 + 100
			
			ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth), n_estimators = trees)
			ABC.fit(trainHoG, trainLabel)
			print "Fitting Done for max depth =",max_depth,"and trees =",trees
			
			ada_boost_accuracy[i,j] = ABC.score(validationHoG, validationLabel)*100.0
			print "Accuracy:",ada_boost_accuracy[i,j]
			
			if ada_boost_accuracy[i,j]> best_accuracy:
				best_accuracy = ada_boost_accuracy[i,j]
				best_depth = max_depth
				best_trees = trees

	np.savetxt("adaboost_training_stats.txt",ada_boost_accuracy,fmt = '%10.5f')
	'''
	print "Finding accuracy for parameters:"
	print "Number of estimators:",best_trees
	print "Depth of trees:", best_depth
	ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth = best_depth), n_estimators = best_trees)
	ABC.fit(trainHoG, trainLabel)
	print "Accuracy:", ABC.score(testHoG, testLabel)*100.0
