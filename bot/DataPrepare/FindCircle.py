# -*- coding:utf-8 -*-
'''
搜寻图像中的圆圈
'''

import sys
import cv2
import math
import numpy
from scipy.ndimage import label
import video

pi_4 = 4*math.pi

def segment_on_dt(img):
    border = img - cv2.erode(img, None)

    dt = cv2.distanceTransform(255 - img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)

    lbl, ncc = label(dt)
    lbl[border == 255] = ncc + 1

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), lbl)
    lbl[lbl < 1] = 0
    lbl[lbl > ncc] = 0

    lbl = lbl.astype(numpy.uint8)
    lbl = cv2.erode(lbl, None)
    lbl[lbl != 0] = 255
    return lbl


def find_circles(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 2)

    edges = frame_gray - cv2.erode(frame_gray, None)
    _, bin_edge = cv2.threshold(edges, 0, 255, cv2.THRESH_OTSU)
    height, width = bin_edge.shape
    mask = numpy.zeros((height+2, width+2), dtype=numpy.uint8)
    cv2.floodFill(bin_edge, mask, (0, 0), 255)

    components = segment_on_dt(bin_edge)

    circles, obj_center = [], []
    contours, _ = cv2.findContours(components,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        c = c.astype(numpy.int64) # XXX OpenCV bug.
        area = cv2.contourArea(c)
        if 100 < area < 3000:
            arclen = cv2.arcLength(c, True)
            circularity = (pi_4 * area) / (arclen * arclen)
            if circularity > 0.5: # XXX Yes, pretty low threshold.
                circles.append(c)
                box = cv2.boundingRect(c)
                obj_center.append((box[0] + (box[2] / 2), box[1] + (box[3] / 2)))

    return circles, obj_center

def track_center(objcenter, newdata):
    for i in xrange(len(objcenter)):
        ostr, oc = objcenter[i]
        best = min((abs(c[0]-oc[0])**2+abs(c[1]-oc[1])**2, j)
                for j, c in enumerate(newdata))
        j = best[1]
        if i == j:
            objcenter[i] = (ostr, new_center[j])
        else:
            print "Swapping %s <-> %s" % ((i, objcenter[i]), (j, objcenter[j]))
            objcenter[i], objcenter[j] = objcenter[j], objcenter[i]


video = video.create_capture(0)

obj_center = None
while True:
    ret, frame = video.read()
    if not ret:
        break

    circles, new_center = find_circles(frame)
    if obj_center is None:
        obj_center = [(str(i + 1), c) for i, c in enumerate(new_center)]
    else:
        track_center(obj_center, new_center)

    for i in xrange(len(circles)):
        cv2.drawContours(frame, circles, i, (0, 255, 0))
        cstr, ccenter = obj_center[i]
        cv2.putText(frame, cstr, ccenter, cv2.FONT_HERSHEY_COMPLEX, 0.5,
                (255, 255, 255), 1, cv2.CV_AA)

    cv2.imshow("result", frame)
    cv2.waitKey(10)
    if len(circles[0]) < 5:
        print "lost something"