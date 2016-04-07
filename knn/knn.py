import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from PIL import Image, ImageOps
from sklearn.neighbors import KNeighborsClassifier

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

def constructHoG(data, orientations = 10, cell_size = (4,4), block_size = (2,2) ):
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
	
	knn_accuracy = np.zeros(shape = (51,2))
	K = [1,2,3,4,5,6,7,8,9,10]
	best_k = -1
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

	print "Finding accuracy for best found parameters:"
	print "Best k:",best_k
	KNN = KNeighborsClassifier(n_neighbors = best_k, weights = 'distance', metric = 'euclidean')
	KNN.fit(trainHoG,trainLabel)
	print "Accuracy:", KNN.score(testHoG, testLabel)*100.0
