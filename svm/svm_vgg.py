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
def featureextract(it, featurevector):
	training_x = []
	training_y = []

	test_x = []
	test_y = []

	with open(featurevector,'r') as f:
		lines = f.readlines()
	num = len(lines)/3


	objects = ["auto", "car", "bicycle", "motorcycle", "person", "rickshaw"]

	for i in range(0, num):
		objtype = objects.index(lines[3*i].split('/')[5])
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

def getAccuracy(svm,data,true_label):
	predicted_label = svm.predict(data)
	accuracy = np.zeros(shape = (7,2))
	for i in range(len(true_label)):
		accuracy[true_label[i],1]+=1
		if true_label[i] == predicted_label[i]:
			accuracy[true_label[i],0]+=1

	accuracy[6] = np.sum(accuracy[0:6], axis = 0)
	return svm.score(data,true_label)*100.0, accuracy

if __name__ == '__main__':
	accuracy = np.zeros(shape = (7,2))
	svm_accuracy = np.zeros(shape = (N+1))
	for it in range(N):
		trainData, trainLabel, testData,testLabel = featureextract(it, sys.argv[1])	
		# print "Data Loaded..."

		SVM = LinearSVC(loss = 'hinge', penalty = 'l2')
		SVM.fit(trainData,trainLabel)
		svm_accuracy[it], temp = getAccuracy(SVM, testData, testLabel)
		accuracy += temp
		# SVM.score(testData, testLabel)*100.0
		print it+1,"--> Predictions:",len(testLabel),"--> Accuracy:",svm_accuracy[it]
	svm_accuracy[N] = np.mean(svm_accuracy[:N])

	for i in range(7):
		accuracy[i,0] = accuracy[i,0]*100.0/accuracy[i,1]
	print accuracy[:,0]
	print "Average accuracy:",svm_accuracy[N]
	# np.savetxt("svm_large_data_night_"+ sys.argv[1][3:] + "_" +str(N)+"fold.txt",svm_accuracy,fmt = '%10.5f')