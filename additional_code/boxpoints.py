import cv2
import numpy as np
import math
import glob


def findCorners(contour):
    """blank_image = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    cv2.drawContours(blank_image, contour, -1, (255, 255, 255))
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-45,0.5)
    dst = cv2.warpAffine(blank_image,M,(cols,rows))
    cv2.imshow("rotatio", dst)
    cv2.waitKey()"""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    print(box)
    print(type(box))
    box = np.int0(box)
    print(box)
    print(type(box))
    height_px_1 = box[0][1] - box[3][1]
    height_px_2 = box[1][1] - box[2][1]
    print (height_px_1, height_px_2)
    if height_px_1 < height_px_2:
        close_height_px = height_px_2
        far_height_px = height_px_1
    else:
        close_height_px = height_px_1
        far_height_px = height_px_2

    return close_height_px, far_height_px

image = cv2.imread("skewed_ku.png")
im2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

_, contours, hierarchy = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
a = findCorners(contours[0])
