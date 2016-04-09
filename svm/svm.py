import numpy as np
import os, sys, string, struct
from array import array
from skimage.feature import hog
from sklearn.svm import LinearSVC
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
			# print obj_dir + "/" + file
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

def getAccuracy(svm,data,true_label):
	predicted_label = svm.predict(data)
	accuracy = np.zeros(shape = (7,2))
	for i in range(len(true_label)):
		accuracy[true_label[i],1]+=1
		if true_label[i] == predicted_label[i]:
			accuracy[true_label[i],0]+=1

	accuracy[6] = np.sum(accuracy[0:6], axis = 0)
	# print accuracy[:,0]
	return svm.score(data,true_label)*100.0, accuracy


if __name__ == '__main__':
	totalData, totalLabel = getData("day")
	accuracy = np.zeros(shape = (7,2))
	best_orientation = 10
	best_cell = 4
	print "Data Loaded..."
	totalData = constructHoG(totalData, orientations = best_orientation, cell_size = (best_cell,best_cell))
	print "Constructed HoG..."
	'''
	trainData, trainLabel = getData("train")
	validationData, validationLabel = getData("validation")
	testData, testLabel = getData("test")
	testData+=validationData
	testLabel+=validationLabel

	svm_accuracy = np.zeros(shape = (16,16)) #from 3 to 18
	
	best_orientation = 10
	best_cell = 4
	best_accuracy = -1
	
	for i in range(16):
		o = i+3
		for j in range(16):
			c = j+3
			trainHoG = constructHoG(trainData, orientations = o, cell_size = (c,c))
			validationHoG = constructHoG(validationData, orientations = o, cell_size = (c,c))
			print "Constructed HoG for orientation =",o,"and cell_size =",(c,c)
			SVM = LinearSVC(loss = 'hinge', penalty = 'l2')
			SVM.fit(trainHoG,trainLabel)
			svm_accuracy[i,j] = SVM.score(validationHoG, validationLabel)*100.0
			print svm_accuracy[i,j]
			if svm_accuracy[i,j] > best_accuracy:
				best_accuracy = svm_accuracy[i,j]
				best_orientation = o
				best_cell = c 
	np.savetxt("svm_training_stats.txt",svm_accuracy,fmt = '%10.5f')
	'''

	print "Finding SVM accuracy for parameters:"
	print "Cell size:",best_cell
	print "Orientations:", best_orientation


	# testHoG = constructHoG(testData, orientations = best_orientation, cell_size = (best_cell,best_cell))
	
	svm_accuracy = np.zeros(shape = (N+1))
	for it in range(N):
		trainData, trainLabel, testData, testLabel = featureextract(totalData, totalLabel, it)	
		print "Data Loaded for",it+1,"iteration..."

		SVM = LinearSVC(loss = 'hinge', penalty = 'l2')
		SVM.fit(trainData,trainLabel)
		svm_accuracy[it],temp = getAccuracy(SVM,testData,testLabel)
		accuracy+=temp
		print it+1,"--> Predictions:",len(testLabel),"--> Accuracy:",svm_accuracy[it]
	svm_accuracy[N] = np.mean(svm_accuracy[:N])
	print "Average accuracy:",svm_accuracy[N]
	for i in range(7):
		accuracy[i,0] = accuracy[i,0]*100.0/accuracy[i,1]
	print accuracy[:,0] 
	# np.savetxt("svm_hog_large_data_night_"+ str(N)+"fold.txt",svm_accuracy,fmt = '%10.5f')