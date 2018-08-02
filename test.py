# import cv2
# import numpy as np
#
#
# def largest_contour(contours):
#     """return index of contour with largest area"""
#
#     area = 0
#     index = 0
#     for idx, cnt in enumerate(contours):
#         tmp = cv2.contourArea(cnt)
#         if tmp > area:
#             area = tmp
#             index = idx
#
#     return index
#
# img = cv2.imread('dataset/Train/6/324.jpg')
# height, width = img.shape[:2]
# im = img[height / 2 - 48: height / 2 + 48, height / 2 - 48: height / 2 + 48]
# im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl = clahe.apply(im)
# ret, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#
# cv2.imshow('th', th)
# cv2.imshow('closing', closing)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import os
import cv2
import joblib


# Generate testing set
TEST_PATH = "dataset\Test"
clf = joblib.load('classifier_th.pkl')
kernel = np.ones((3, 3), np.uint8)
list_folder = os.listdir(TEST_PATH)
testset = []
test_label = []
for folder in list_folder:
    flist = os.listdir(os.path.join(TEST_PATH, folder))
    for f in flist:
        im = cv2.imread(os.path.join(TEST_PATH, folder, f))
        height, width = im.shape[:2]
        im = im[height/2 - 48 : height/2 + 48, height/2 - 48 : height/2 + 48]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(im)
        ret, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        testset.append(closing)
        test_label.append(int(folder))

# Testing
print("Testing start")
testset = np.reshape(testset, (len(testset), -1))
y = clf.predict(testset)
print("Testing accuracy: " + str(clf.score(testset, test_label)))