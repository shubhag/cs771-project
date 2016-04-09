import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from PIL import Image, ImageOps
import cPickle

N = 5

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
	best_orientation = 3
	best_size = 160
	best_cell = 7
	print "Data Loaded..."
	totalData = constructHoG(totalData, orientations = best_orientation, cell_size = (best_cell,best_cell))
	print "Constructed HoG..."
	'''
	trainData, trainLabel = getData("train")
	validationData, validationLabel = getData("validation")
	testData, testLabel = getData("test")
	testData += validationData
	testLabel += validationLabel
	
	print "Data Loaded..."

	forest_accuracy = np.zeros(shape = (8,8,20))
	
	best_size = 160
	best_accuracy = -1
	best_orientation = 3
	best_cell = 7
	
	for i in range(8):
		c = i+3
		for j in range(8):
			o = j+3
			trainHoG = constructHoG(trainData, orientations = o, cell_size = (c,c))
			validationHoG = constructHoG(validationData, orientations = o, cell_size = (c,c))
			print "Constructed HoG for cell_size:",(c,c),"and orientations:",o
			for k in range(20):
				t = (k+1)*20
				RFC = RandomForestClassifier(n_estimators = t)
				RFC.fit(trainHoG,trainLabel)
				forest_accuracy[i,j,k] = RFC.score(validationHoG, validationLabel)*100.0
				print "Trees:",t,"Accuracy: ",forest_accuracy[i,j,k]
				if forest_accuracy[i,j,k] > best_accuracy:
					best_accuracy = forest_accuracy[i,j,k]
					best_size = t
					best_orientation = o
					best_cell = c		

	cPickle.dump( forest_accuracy, open( "forest_training_stats.pkl", "wb" ))
	'''
	print "Finding accuracy for parameters:"
	print "Tree size:", best_size
	print "Orientation:", best_orientation
	print "Cell size:", best_cell

	forest_accuracy = np.zeros(shape = (N+1))
	for it in range(N):
		trainData, trainLabel, testData, testLabel = featureextract(totalData, totalLabel, it)	
		print "Data Loaded for",it+1,"iteration..."
		RFC = RandomForestClassifier(n_estimators = best_size)
		RFC.fit(trainData,trainLabel)
		forest_accuracy[it] = RFC.score(testData, testLabel)*100.0

		print it+1,"--> Predictions:",len(testLabel),"--> Accuracy:",forest_accuracy[it]
	forest_accuracy[N] = np.mean(forest_accuracy[:N])
	print "Average accuracy:",forest_accuracy[N]
	np.savetxt("forest_hog_large_data_night_"+ str(N)+"fold.txt",forest_accuracy,fmt = '%10.5f')