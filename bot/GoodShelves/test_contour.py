# -*- coding:utf-8 -*-
'''
  测试 图像轮廓坐标
  http://blog.csdn.net/sunny2038/article/details/12889059

'''

import cv2
import copy

img = cv2.imread('E:\\bot_GoodsShelves\\data\\test_out\\wind.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print gray
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# cv2.findContours()函数首先返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示。
hierarchy,contours,index = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

del_index =[]
cont_copy =[]
m = 0
# for j in range(len(contours)):
#     if len(contours[j])>4:
#         cont_copy.append(contours[j])
#         print ' add del index %d   '%j
#         m+=1


cv2.drawContours(img,cont_copy,-1,(0,255,0),3)
cv2.imshow("img", img)
cv2.waitKey(0)