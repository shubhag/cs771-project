import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from PIL import Image, ImageOps
from sklearn.neighbors import KNeighborsClassifier

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

# def constructHoG(data, orientations = 10, cell_size = (4,4), block_size = (2,2) ):
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
	trainData, trainLabel,validationData, validationLabel,testData, testLabel = featureextract()
	testData+=validationData
	testLabel+=validationLabel
	print "Data Loaded..."
	
	knn_accuracy = np.zeros(shape = (51,2))
	K = [1,2,3,4,5,6,7,8,9,10]
	best_k = 3
	best_accuracy = -1
	'''
	for i in range(51):
		if i<10:
			k = K[i]
		else:
			k = i*3
		knn_accuracy[i,0] = k
		KNN = KNeighborsClassifier(n_neighbors = k, weights = 'distance', metric = 'euclidean')
		KNN.fit(trainData,trainLabel)
		knn_accuracy[i,1] = KNN.score(validationData, validationLabel)*100.0
		if best_accuracy < knn_accuracy[i,1]:
			best_accuracy = knn_accuracy[i,1]
			best_k = k 
		print "k:",k,"Accuracy:",knn_accuracy[i,1]
	np.savetxt("knn_vgg_training_stats.txt",knn_accuracy,fmt = '%10.5f')
	'''
	print "Finding accuracy for parameters:"
	print "K:",best_k
	KNN = KNeighborsClassifier(n_neighbors = best_k, weights = 'distance', metric = 'euclidean')
	KNN.fit(trainData,trainLabel)
	print "Accuracy:", KNN.score(testData, testLabel)*100.0
