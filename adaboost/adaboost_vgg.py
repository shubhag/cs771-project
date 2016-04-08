import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image, ImageOps

# def getData(folder):
# 	FV = []
# 	FL = []
# 	cur_dir = os.path.dirname(os.path.abspath(__file__))
# 	cur_dir = os.path.join(cur_dir, "makedataset")
# 	folders = ["auto", "car", "bicycle", "motorcycle", "person"]
# 	obj_id = [0, 1, 2, 3, 4]
# 	size = (36, 36)

# 	data_dir = os.path.join(cur_dir, folder)
# 	for i in range(0,5):
# 		folder = folders[i]
# 		obj_dir = os.path.join(data_dir, folder)
# 		files = os.listdir(obj_dir)
# 		for file in files:
# 			col = Image.open(obj_dir+"/"+file)
# 			gray = col.convert('L')
# 			gray.thumbnail(size, Image.ANTIALIAS)
# 			background = Image.new('L', size)
# 			background.paste( gray, ((size[0] - gray.size[0]) / 2, (size[1] - gray.size[1]) / 2))
			
# 			bw = np.asarray(background)
# 			FV.append(bw)
# 			FL.append(obj_id[i])

# 	FV = np.asarray(FV)
# 	FL = np.asarray(FL)
# 	# FV = FV.reshape((FV.shape[0], 1, size[0], size[1]))
# 	print FV.shape
# 	return FV,FL

# def constructHoG(data, orientations = 3, cell_size = (7,7), block_size = (2,2) ):
# 	res = []
# 	for img in data:
# 		FV = hog(img, orientations, cell_size, block_size)
# 		res.append(FV)
# 	return np.asarray(res)

def featureextract():
	training_x = []
	training_y = []

	test_x = []
	test_y = []

	validation_x = []
	validation_y = []

	with open('../../featurevectorvgg-f7.txt','r') as f:
		lines = f.readlines()
	num = len(lines)/3

	objects = ["auto", "car", "bicycle", "motorcycle", "person", "rickshaw"]

	for i in range(0, num):
		objtype = objects.index(lines[3*i].split('/')[4])
		if objtype >= 0 :
			vector = np.array(map(float, lines[3*i + 1].strip().split(' ')))
			if i%10 == 0:
				validation_x.append(vector)
				validation_y.append(objtype)
			elif i%10 ==1:
				test_x.append(vector)
				test_y.append(objtype)
			else:
				training_x.append(vector)
				training_y.append(objtype)
		else:
			print objtype
			print lines[3*i]

	return training_x, training_y, validation_x, validation_y, test_x, test_y

if __name__ == '__main__':
	trainData, trainLabel, validationData, validationLabel, testData, testLabel = featureextract()
	print "Data Loaded..."
	print "Dimension:",len(trainData[0])
	ada_boost_accuracy = np.zeros(shape = (5,10))
	print "Adaboost Classifiction"
	best_depth = 8
	best_trees = 240
	'''
	best_accuracy = -1

	for i in range(5):
		max_depth = (i+1)*4
		for j in range(10):
			trees = (j+1)*20 + 100
			
			ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth), n_estimators = trees)
			ABC.fit(trainData, trainLabel)
			print "Fitting Done for max depth =",max_depth,"and trees =",trees
			
			ada_boost_accuracy[i,j] = ABC.score(validationData, validationLabel)*100.0
			print "Accuracy:",ada_boost_accuracy[i,j]
			
			if ada_boost_accuracy[i,j]> best_accuracy:
				best_accuracy = ada_boost_accuracy[i,j]
				best_depth = max_depth
				best_trees = trees

	np.savetxt("adaboost_vgg_training_stats.txt",ada_boost_accuracy,fmt = '%10.5f')
	'''
	print "Finding accuracy for parameters:"
	print "Number of estimators:",best_trees
	print "Depth of trees:", best_depth
	ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth = best_depth), n_estimators = best_trees)
	ABC.fit(trainData, trainLabel)	
	print "Accuracy:", ABC.score(testData, testLabel)*100.0