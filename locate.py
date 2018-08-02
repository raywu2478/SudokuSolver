import cv2
import numpy as np


def largest_contour(contours):
    """return index of contour with largest area"""

    area = 0
    index = 0
    for idx, cnt in enumerate(contours):
        tmp = cv2.contourArea(cnt)
        if tmp > area:
            area = tmp
            index = idx

    return index


def find_corners(pts):
    """return list of coordinates such that
    first entry in the list is the top-left,
    second entry is the top-right,
    third is bottom-right,
    fourth is the bottom-left"""

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")


    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=2)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=2)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def find_pizzle(im):
    height, width = im.shape[:2]
    img = cv2.resize(im, (width / 2, height / 2), interpolation=cv2.INTER_CUBIC)

    #blur = cv2.medianBlur(img, 5)
    #gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    th = cv2.adaptiveThreshold(cl.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 401, 6)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    im2, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(img, contours, -1, (0,255,0), 3)#----------------------------------------------
    #cv2.drawContours(img, contours, largest_contour(contours), (0, 255, 255),3)  # -----------------
    #cv2.namedWindow('closing', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('closing', width / 4, height / 4)
    #cv2.imshow('closing', closing)

    cnt = contours[largest_contour(contours)]

    hull = cv2.convexHull(cnt)

    epsilon = 0.1*cv2.arcLength(hull,True)
    approx = cv2.approxPolyDP(hull,epsilon,True)

    # if len(approx) != 4:
    #     raise ValueError("Bounding square has ", len(approx), " points")

    rect = find_corners(approx)

    #compute width and height of new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    #set largest width and height to image
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    maxSquare = max(int(maxHeight), int(maxWidth))

    #create destination roi
    dst = np.array([
        [0, 0],
        [maxSquare - 1, 0],
        [maxSquare - 1, maxSquare - 1],
        [0, maxSquare - 1]], dtype="float32")

    #warp perspective
    M = cv2.getPerspectiveTransform(rect, dst)
    #return cv2.warpPerspective(closing, M, (maxSquare, maxSquare))
    return cv2.warpPerspective(img, M, (maxSquare, maxSquare))

