
# echo "Start SVM"

# python svm/svm.py
# echo "Done HoG"
# python svm/svm_vgg.py night_alexnet.txt
# echo "Done Alex Net"
# python svm/svm_vgg.py night_vgg_cnn_f.txt
# echo "Done VGG F"
# python svm/svm_vgg.py night_vgg_cnn_s.txt
# echo "Done VGG S"

# echo "Start KNN"

# python knn/knn.py
# echo "Done HoG"
# python knn/knn_vgg.py night_alexnet.txt
# echo "Done Alex Net"
# python knn/knn_vgg.py night_vgg_cnn_f.txt
# echo "Done VGG F"
# python knn/knn_vgg.py night_vgg_cnn_s.txt
# echo "Done VGG S"

# echo "Start random forest"

# python forest/forest.py
# echo "Done HoG"
# python forest/forest_vgg.py night_alexnet.txt
# echo "Done Alex Net"
# python forest/forest_vgg.py night_vgg_cnn_f.txt
# echo "Done VGG F"
# python forest/forest_vgg.py night_vgg_cnn_s.txt
# echo "Done VGG S"

echo "Start Adaboost"

python adaboost/adaboost.py
echo "Done HoG"
python adaboost/adaboost_vgg.py night_alexnet.txt
echo "Done Alex Net"
python adaboost/adaboost_vgg.py night_vgg_cnn_f.txt
echo "Done VGG F"
python adaboost/adaboost_vgg.py night_vgg_cnn_s.txt
echo "Done VGG S"


