import cv2
import numpy as np
import math

def overlap(a,b,c,d):
    if ((a - c) * (d - b) > 0):
        return True
    elif ((d - a) * (b - c) > 0) and min(abs(a-d),abs(b-c)) > 0.25* max(abs(b-a), abs(d-c)):
        return True
    else:
        return False

def detect_digits(indx):
    strip = strips[indx]

    gray_strip = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_strip, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 3, 2)

    _, contours_strip, hierarchy_strip = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    #Introduce lists of left x1, right x2, top y1, bottom pixels y2 without drawing rectangles
    x1, x2, y1, y2 = ([] for i in range(4))
    for cnt in contours_strip:
        x, y, w, h = cv2.boundingRect(cnt)
        x1.append(x)
        x2.append(x+w)
        y1.append(y)
        y2.append(y+h)
        # cv2.rectangle(strip, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # digits.append(gray_strip[y:y + h, x:x + w])

    #Introduce lists for future rectangles
    xx, xw, yy, yh = ([] for i in range(4))
    flags = [False]*len(x1)
    current_flags = [False]*len(x1)
    #Search for pairs of intersecting in width(!) contours
    for i in range(len(x1)):
        if not flags[i]:
            j = i + 1
            while (j < len(x1)):
                if overlap(x1[i], x2[i], x1[j], x2[j]):
                    # xx.append(min(x1[i],x1[j]))
                    # xw.append(max(x2[i],x2[j]))
                    # yy.append(min(y1[i], y1[j]))
                    # yh.append(max(y2[i], y2[j]))
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

    #Draw the rectangles from those validated lists xx,xw,yy,yh
    for i in range(len(xx)):
        if (xx[i]-xw[i])*(yy[i]-yh[i])>10:
            cv2.rectangle(strip, (xx[i], yy[i]), (xw[i], yh[i]), (0, 0, 255), 1)
            digits.append(gray_strip[yy[i]:yh[i], xx[i]:xw[i]])

    height, width = strip.shape[:2]
    strip_large = cv2.resize(strip, (2 * width, 2 * height))
    cv2.imshow('detected digits', strip_large)
    cv2.waitKey(0)

    # digit = digits[-6]
    # height, width = digit.shape[:2]
    # digit_large = cv2.resize(digit, (10 * width, 10 * height))
    # cv2.imshow('detected digits', digit_large)
    # cv2.waitKey(0)


def detect_lines(fn):
    src = cv2.imread(fn)
    im = cv2.Canny(src, 50, 200, L2gradient=True)
    im1 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(im, 1, math.pi / 180.0, 30, np.array([]), 80, 20)
    a, b, c = lines.shape
    for i in range(a):
        cv2.line(im1, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 0, 0), 4, cv2.LINE_AA)
    #
    # cv2.imshow("detected lines", im1)
    # cv2.waitKey(0)

    im2 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

    _, contours, hierarchy = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im2, contours, -1, (127, 0, 127), 1)
    strips = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 220:
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(src, (x,y),(x+w,y+h),(0,255,0),1)
            strips.append(src[y + 1:y + h - 1, x + 1:x + w - 1])
    return strips

    # cv2.imshow('detected contours',src)
    # cv2.waitKey(0)


fn = "poem.jpg"
strips = detect_lines(fn)

for i in range(len(strips)):
    detect_digits(i)



