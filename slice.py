import cv2
import random
import numpy as np
import joblib


clf = joblib.load('classifier_th.pkl')

cv2.namedWindow('X', cv2.WINDOW_NORMAL)
cv2.resizeWindow('X', 100, 100)

def gather(puzzle):
    points = []
    height, width = puzzle.shape[:2]
    for i in range(10):
        for j in range(10):
            points.append((j * (width/9), i * (height / 9)))

    for i in range(0, 9):
        for j in range(0, 9):
            y1 = int(points[j + i * 10][1] + 5)
            y2 = int(points[j + i * 10 + 11][1] - 5)
            x1 = int(points[j + i * 10][0] + 5)
            x2 = int(points[j + i * 10 + 11][0] - 5)

            #cv2.imwrite(str(i) + str(j) + str(random.randint(1,101)) + '.jpg', puzzle[y1: y2,x1: x2])

            X = puzzle[y1:y2, x1:x2]
            X = cv2.resize(X, (36, 36), interpolation=cv2.INTER_AREA)
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(X)
            ret, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            # cv2.imshow('X', closing)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            num = clf.predict(np.reshape(X, (1, -1)))
            if (num[0] != 0):
                cv2.putText(puzzle, str(num[0]), (int(points[j + i * 10 + 10][0] + 10),int(points[j + i * 10 + 10][1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 0, 0), 3)
            else:
                cv2.putText(puzzle, str(num[0]), (int(points[j + i * 10 + 10][0] + 10),int(points[j + i * 10 + 10][1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 0, 0), 3)

    cv2.namedWindow('warp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('warp', height/2, width/2)
    cv2.imshow('warp', puzzle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()