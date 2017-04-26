from mnist import MNIST
from sklearn.decomposition import dict_learning
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

mndata = MNIST('./data')

tr_images, tr_labels = mndata.load_training()
te_images, te_labels = mndata.load_testing()
n = 1000

train_images = np.array(tr_images)
train_labels = np.array(tr_labels)

test_images = np.array(te_images)
test_labels = np.array(te_labels)

# print(test_images.shape)
# svm = SVC()
# svm.fit(train_images[:n, :], train_labels[:n])
# pred_labels = svm.predict(test_images)
# print("SVM Accuracy:", sum(test_labels == pred_labels) / len(pred_labels))

# knn = KNeighborsClassifier()
# knn.fit(train_images[:n, :], train_labels[:n])
# pred_labels = svm.predict(test_images)
# print("KNN Accuracy:"sum(test_labels == pred_labels) / len(pred_labels))

U, W = dict_learning(train_images[:n, :], 1024, 0.1)
print(U.shape)

# knn = KNeighborsClassifier()
# knn.fit(train_images[:n, :], train_labels[:n])
# pred_labels = svm.predict(test_images)
# print("KNN Accuracy:"sum(test_labels == pred_labels) / len(pred_labels))

# svm.fit(train_images, )