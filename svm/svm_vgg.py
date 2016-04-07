import numpy as np
import os, sys, string, struct
from array import array
from sklearn.svm import LinearSVC
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
N = 5
def featureextract(it):
	training_x = []
	training_y = []

	test_x = []
	test_y = []

	with open('../../featurevectorvgg-f7.txt','r') as f:
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
	svm_accuracy = np.zeros(shape = (N+1))
	for it in range(1,N+1):
		trainData, trainLabel, testData,testLabel = featureextract(it-1)	
		print "Data Loaded..."

		SVM = LinearSVC(loss = 'hinge')
		SVM.fit(trainData,trainLabel)
		svm_accuracy[it-1] = SVM.score(testData, testLabel)*100.0
		print it,"--> Predictions:",len(testLabel),"Accuracy:",svm_accuracy[it-1]
	svm_accuracy[N] = np.mean(svm_accuracy[:N])
	np.savetxt("svm_vggnet_results_"+ str(N)+"fold.txt",svm_accuracy,fmt = '%10.5f')