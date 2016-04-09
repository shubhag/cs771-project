import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image, ImageOps

N = 5

def getData(fd):
	FV = []
	FL = []
	cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	cur_dir = os.path.join(cur_dir, "makedataset")
	data_dir = os.path.join(cur_dir, fd)
	folders = ["auto", "car", "bicycle", "motorcycle", "person", "rickshaw"]
	obj_id = [0, 1, 2, 3, 4, 5]
	size = (36, 36)

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
	return FV,FL

def featureextract(FV, FL, it):
	training_x = []
	training_y = []

	test_x = []
	test_y = []

	for i in range(len(FL)):
		if i%N == it:
			test_x.append(FV[i])
			test_y.append(FL[i])
		else:
			training_x.append(FV[i])
			training_y.append(FL[i])
	return training_x, training_y, test_x, test_y

def constructHoG(data, orientations = 3, cell_size = (7,7), block_size = (2,2) ):
	res = []
	for img in data:
		FV = hog(img, orientations, cell_size, block_size)
		res.append(FV)
	return np.asarray(res)

if __name__ == '__main__':
	totalData, totalLabel = getData("night")
	best_trees = 140
	best_depth = 12
	print "Data Loaded..."
	totalData = constructHoG(totalData)
	print "Constructed HoG..."
	'''
	trainData, trainLabel = getData("train")
	validationData, validationLabel = getData("validation")
	testData, testLabel = getData("test")
	testData+=validationData
	testLabel+=validationLabel
	print "Data Loaded..."

	trainHoG = constructHoG(trainData)
	validationHoG = constructHoG(validationData)
	testHoG = constructHoG(testData)
	
	ada_boost_accuracy = np.zeros(shape = (10,10))
	print "Adaboost Classifiction"
	best_depth = 12
	best_trees = 140
	
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
	
	adaboost_accuracy = np.zeros(shape = (N+1))
	for it in range(N):
		trainData, trainLabel, testData, testLabel = featureextract(totalData, totalLabel, it)	
		print "Data Loaded for",it+1,"iteration..."

		ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth = best_depth), n_estimators = best_trees)
		ABC.fit(trainData,trainLabel)
		adaboost_accuracy[it] = ABC.score(testData, testLabel)*100.0
		print it+1,"--> Predictions:",len(testLabel),"--> Accuracy:",adaboost_accuracy[it]
	adaboost_accuracy[N] = np.mean(adaboost_accuracy[:N])
	print "Average accuracy:",adaboost_accuracy[N]
	np.savetxt("adaboost_hog_large_data_night_"+ str(N)+"fold.txt",adaboost_accuracy,fmt = '%10.5f')