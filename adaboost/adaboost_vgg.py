import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image, ImageOps
import sys
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

N = 5
def featureextract(it, featurevector):
	print featurevector
	training_x = []
	training_y = []

	test_x = []
	test_y = []

	# with open('../../featurevectorvgg-f7.txt','r') as f:
	with open(featurevector,'r') as f:
		lines = f.readlines()
	num = len(lines)/3


	objects = ["auto", "car", "bicycle", "motorcycle", "person", "rickshaw"]

	for i in range(0, num):
		objtype = objects.index(lines[3*i].split('/')[4])
		if objtype >= 0 :
			vector = np.array(map(float, lines[3*i + 1].strip().split(' ')))
			if i%N ==it:
				test_x.append(vector)
				test_y.append(objtype)
			else:
				training_x.append(vector)
				training_y.append(objtype)
		else:
			print objtype
			print lines[3*i]

	return training_x, training_y, test_x, test_y

if __name__ == '__main__':
	adaboost_accuracy = np.zeros(shape = (N+1))
	best_depth = 8
	best_trees = 240
	print "Finding accuracy for parameters:"
	print "Number of estimators:",best_trees
	print "Depth of trees:", best_depth

	for it in range(N):
		trainData, trainLabel, testData,testLabel = featureextract(it, sys.argv[1])	
		print "Data Loaded for",it+1,"iteration..."

		ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth = best_depth), n_estimators = best_trees)
		ABC.fit(trainData,trainLabel)
		adaboost_accuracy[it] = ABC.score(testData, testLabel)*100.0
		print it+1,"--> Predictions:",len(testLabel),"--> Accuracy:",adaboost_accuracy[it]
	adaboost_accuracy[N] = np.mean(adaboost_accuracy[:N])
	print "Average accuracy:",adaboost_accuracy[N]
	np.savetxt("adaboost_large_data_night_" + sys.argv[1][3:] + "_" + str(N)+"fold.txt",adaboost_accuracy,fmt = '%10.5f')
	

