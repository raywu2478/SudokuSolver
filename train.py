import numpy as np
from sklearn.svm import LinearSVC
import os
import cv2
import joblib


TRAIN_PATH = "dataset\Train"
TEST_PATH = "dataset\Test"
kernel = np.ones((3, 3), np.uint8)


def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # cl = clahe.apply(image)
    # ret, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return image


# Generate training set
list_folder = os.listdir(TRAIN_PATH)
trainset = []
kernel = np.ones((3, 3), np.uint8)

#load all data
for folder in list_folder:
    flist = os.listdir(os.path.join(TRAIN_PATH, folder))
    for f in flist:
        im = cv2.imread(os.path.join(TRAIN_PATH, folder, f))
        trainset.append(process_img(im))

# Labeling for trainset
train_label = []
for i in range(0,10):
    temp = 500*[i]
    train_label += temp

trainset = np.reshape(trainset, (5000, -1))

print("Training start")
# Create an linear SVM object
clf = LinearSVC()
# Perform the training
clf.fit(trainset, train_label)
print("Training finished")

# Generate testing set
list_folder = os.listdir(TEST_PATH)
testset = []
test_label = []
for folder in list_folder:
    flist = os.listdir(os.path.join(TEST_PATH, folder))
    for f in flist:
        im = cv2.imread(os.path.join(TEST_PATH, folder, f))
        testset.append(process_img(im))
        test_label.append(int(folder))

# Testing
print("Testing start")
testset = np.reshape(testset, (len(testset), -1))
y = clf.predict(testset)
print("Testing accuracy: " + str(clf.score(testset, test_label)))

joblib.dump(clf, "classifier_gray2.pkl", compress=3)
