import cv2
import numpy as np
import math
import glob

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[-1]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle,1)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nW = int(rows * sin + cols * cos)
    nH = int(rows * cos + cols * sin)
    M[0,2] += nW/2 - cols/2
    M[1,2] += nH/2 - rows/2

    img_rot = cv2.warpAffine(img,M,(nW, nH))

    # rotate bounding box
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    print(pts)
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop

image = cv2.imread("skewed_ku.png")
cv2.imshow('strip', image)
rotated = crop_minAreaRect(image,[120,120,30,30,45])
cv2.imshow('rotated', rotated)
cv2.waitKey(0)