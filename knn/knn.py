import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from PIL import Image, ImageOps
from sklearn.neighbors import KNeighborsClassifier

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

def constructHoG(data, orientations = 10, cell_size = (4,4), block_size = (2,2) ):
	res = []
	for img in data:
		FV = hog(img, orientations, cell_size, block_size)
		res.append(FV)
	return np.asarray(res)

if __name__ == '__main__':
	totalData, totalLabel = getData("night")
	best_orientation = 10
	best_cell = 4
	best_k = 4
	print "Data Loaded..."
	totalData = constructHoG(totalData, orientations = best_orientation, cell_size = (best_cell,best_cell))
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
	
	knn_accuracy = np.zeros(shape = (51,2))
	K = [1,2,3,4,5,6,7,8,9,10]
	best_k = 4
	best_accuracy = -1
	
	for i in range(51):
		if i<10:
			k = K[i]
		else:
			k = i*3
		knn_accuracy[i,0] = k
		KNN = KNeighborsClassifier(n_neighbors = k, weights = 'distance', metric = 'euclidean')
		KNN.fit(trainHoG,trainLabel)
		knn_accuracy[i,1] = KNN.score(validationHoG, validationLabel)*100.0
		if best_accuracy < knn_accuracy[i,1]:
			best_accuracy = knn_accuracy[i,1]
			best_k = k 
		print "k:",k,"Accuracy:",knn_accuracy[i,1]
	np.savetxt("knn_training_stats.txt",knn_accuracy,fmt = '%10.5f')
	'''

	knn_accuracy = np.zeros(shape = (N+1))
	for it in range(N):
		trainData, trainLabel, testData, testLabel = featureextract(totalData, totalLabel, it)	
		print "Data Loaded for",it+1,"iteration..."
		KNN = KNeighborsClassifier(n_neighbors = best_k, weights = 'distance', metric = 'euclidean')
		KNN.fit(trainData,trainLabel)
		knn_accuracy[it] = KNN.score(testData, testLabel)*100.0
		print it+1,"--> Predictions:",len(testLabel),"--> Accuracy:",knn_accuracy[it]
	knn_accuracy[N] = np.mean(knn_accuracy[:N])
	print "Average accuracy:",knn_accuracy[N]
	np.savetxt("knn_hog_large_data_night_"+ str(N)+"fold.txt",knn_accuracy,fmt = '%10.5f')