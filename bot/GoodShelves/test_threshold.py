# -*- coding:utf-8 -*-
'''
  测试 opencv2的阈值函数效果
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r'E:\imgtest\0ce72e376e7c42ed929cfbe28e50e3e8.jpg',0)

# 第一个是原图像(必须为灰度图)，第二个是阈值，第三个叫maxVal，
# 它表示如果这个点的灰度大于(有时是小于，根据第四个输入的设置)阈值，则将这个点的灰度修改成maxVal，第四个参数则决定函数的具体功能
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
thresh5 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
thresh6 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','adaptive_mean','adaptive_gussi']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5,thresh6]

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()