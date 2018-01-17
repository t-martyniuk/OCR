import cv2
import numpy as np
import math
import glob

kernel = np.ones((3,3),np.uint8)
const_of_overlapping = 0.25

def overlap(a, b, c, d):
    if ((a - c) * (d - b) > 0):
        return True
    elif ((d - a) * (b - c) > 0) and min(abs(a - d), abs(b - c)) > const_of_overlapping * max(abs(b - a), abs(d - c)):
        return True
    else:
        return False

def noisy(image):
    row, col, ch = image.shape
    noise = np.random.rand(row, col, ch)
    noise = noise.reshape(row, col, ch)
    return image + image * noise

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
    # print(np.array([box]))
    pts = cv2.transform(box, M)

    # crop
    img_crop = img_rot[pts[3][1]:pts[0][1],
                       pts[3][0]:pts[2][0]]

    return img_crop

def detect_characters(indx):

    strip = strips[indx]
    print(indx)
    cv2.imshow("detected line", strip)
    cv2.waitKey(0)
    # samples = np.empty((0, 100))

    # gray_strip = strip
    gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_strip, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 3, 2)

    cv2.imshow("after thresh", thresh)
    cv2.waitKey(0)


    _, contours_strip, hierarchy_strip = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Introduce lists of left x1, right x2, top y1, bottom pixels y2 without drawing rectangles
    x1, x2, y1, y2 = ([] for i in range(4))
    for cnt in contours_strip:
        x, y, w, h = cv2.boundingRect(cnt)
        x1.append(x)
        x2.append(x + w)
        y1.append(y)
        y2.append(y + h)


    # Introduce lists for future rectangles
    xx, xw, yy, yh = ([] for i in range(4))
    flags = [False] * len(x1)
    current_flags = [False] * len(x1)
    # Search for pairs of intersecting in width(!) contours
    for i in range(len(x1)):
        if not flags[i]:
            j = i + 1
            while (j < len(x1)):
                if overlap(x1[i], x2[i], x1[j], x2[j]):

                    current_flags[i], current_flags[j], flags[i], flags[j] = (True for i in range(4))
                j += 1
            if not flags[i]:
                xx.append(x1[i])
                xw.append(x2[i])
                yy.append(y1[i])
                yh.append(y2[i])
            else:
                minx = min(x1[i] for i in (filter(lambda i: current_flags[i], range(len(x1)))))
                miny = min(y1[i] for i in (filter(lambda i: current_flags[i], range(len(x1)))))
                maxx = max(x2[i] for i in (filter(lambda i: current_flags[i], range(len(x1)))))
                maxy = max(y2[i] for i in (filter(lambda i: current_flags[i], range(len(x1)))))
                xx.append(minx)
                xw.append(maxx)
                yy.append(miny)
                yh.append(maxy)
            current_flags = [False] * len(x1)

    # Draw the rectangles from those validated lists xx,xw,yy,yh
    lenxx = len(xx)
    black_valid = [None]*lenxx
    for i in range(lenxx):
        roi = gray_strip[yy[i]:yh[i], xx[i]:xw[i]]
        _, roi = cv2.threshold(roi,127, maxval = 255, type=cv2.THRESH_BINARY)
        whites = cv2.countNonZero(roi)
        black_valid[i] = whites < 0.9 * (yh[i] - yy[i]) * (xw[i] - xx[i])
        if black_valid[i]:
            roismall = cv2.resize(roi, (100, 100))
            cv2.imshow('character squared', roismall)
            cv2.waitKey(0)
    xx = [xx[i] for i in range(lenxx) if black_valid[i]]
    xw = [xw[i] for i in range(lenxx) if black_valid[i]]
    yy = [yy[i] for i in range(lenxx) if black_valid[i]]
    yh = [yh[i] for i in range(lenxx) if black_valid[i]]

    for i in range(len(xx)):
        cv2.rectangle(strip, (xx[i], yy[i]), (xw[i], yh[i]), (0, 0, 255), 1)

            # samples = np.append(samples, sample, 0)
            # digits.append(gray_strip[yy[i]:yh[i], xx[i]:xw[i]])
    height, width = strip.shape[:2]
    strip_large = cv2.resize(strip, (2 * width, 2 * height))
    cv2.imshow('detected characters', strip_large)
    cv2.waitKey(0)



def detect_lines(src):
    cv2.imshow('start', src)
    cv2.waitKey(0)
    # Convert to gray
    src_grey = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(src_grey, (5, 5), 0)

    # thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 3, 2)
    # cv2.imshow('after adaptive threshold', thresh)
    # cv2.waitKey(0)
    #Reduce noise
    opening = cv2.morphologyEx(src_grey, cv2.MORPH_OPEN, kernel)
    cv2.imshow('after opening', opening)
    cv2.waitKey(0)
    #Black background
    reverted = cv2.bitwise_not(opening)
    src1 = np.uint8(reverted)
    #Canny edge detector
    im = cv2.Canny(src1, 50, 200, L2gradient=True)
    cv2.imshow('after canny', im)
    cv2.waitKey(0)
    #Fill the contours of the letters
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('after closing', im)
    cv2.waitKey(0)
    #Detect lines
    lines = cv2.HoughLinesP(im, 1, math.pi / 180.0, 60, np.array([]), 80, 20)
    a, b, c = lines.shape
    #Convert to coloured to draw blue(currently) lines
    im1 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    #Draw the lines
    for i in range(a):
        cv2.line(im1, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 0, 0), 4, cv2.LINE_AA)

    cv2.imshow("detected lines", im1)
    cv2.waitKey(0)
    #Convert back to gray
    im2 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    #Detect contours of the lines
    _, contours, hierarchy = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Draw the contours (thin gray)
    cv2.drawContours(im2, contours, -1, (127, 0, 127), 1)
    strips = []
    cv2.imshow("detected lines&contours", im2)
    cv2.waitKey(0)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(src, (x,y),(x+w,y+h),(0,255,0),1)
        if (h > 2) and (w > 2) and (cv2.contourArea(cnt) > 220):
            strips.append(src[y + 1:y + h - 1, x + 1:x + w - 1])
    # #Detect skewed bounding rectangles
    # len_angles = len(contours)
    # angles = [cv2.minAreaRect(contours[i])[-1] for i in range(len_angles)]
    # mean = np.mean(np.array(angles))
    # sd = np.std(np.array(angles))
    # final_contours = [contours[i] for i in range(len_angles) if abs(angles[i] - mean) < 2 * sd]
    # final_angles = [angles[i] for i in range(len_angles) if abs(angles[i] - mean) < 2 * sd ]
    #
    # for cnt in final_contours:
    #     print(cv2.minAreaRect(cnt))
    #     strip = crop_minAreaRect(src_grey,cv2.minAreaRect(cnt))
    #     # if (strip.shape[0] > 0) and (strip.shape[1] > 0):
    #     #     strips.append(strip)

        # if cv2.contourArea(cnt) > 220:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     # cv2.rectangle(src, (x,y),(x+w,y+h),(0,255,0),1)
        #     strips.append(src[y + 1:y + h - 1, x + 1:x + w - 1])
    return strips

    # cv2.imshow('detected contours',src)
    # cv2.waitKey(0)
#images = glob.glob("image_samples_with_fonts/train/*.bmp")
#images = glob.glob("*.JPG")
images = glob.glob("pictures/polska.jpg")

#For all images
for img in images:
    src = cv2.imread(img)
    #Detect the lines in the image
    strips = detect_lines(src)
    print(len(strips))
    for strip in strips:
        cv2.imshow('strip', strip)
        cv2.waitKey(0)

    #For all lines
    for i in range(len(strips)):
        #Detect the squared characters
        detect_characters(i)
