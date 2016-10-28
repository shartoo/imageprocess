# -*- coding:utf-8 -*-
'''

Assuming you ran cv2.findContours on your image, you will have received a structure that lists all
of the contours available in your image. I'm also assuming that you know the index of the contour that
was used to surround the object you want. Assuming this is stored in idx, first use cv2.drawContours
to draw a filled version of this contour onto a blank image, then use this image to index into your image
to extract out the object. So something like this, assuming your image is a grayscale image stored in img:

'''

import numpy as np
import cv2
from imtool import *
from point import point

imgfile =r'E:\BOT_Image_SecondRound_Supermarket_Train_fj394hx7 (1)\Train\Instant Noodle\0ccec8992d654788a7140583b4a8e545.jpg'
#imgfile =r'E:\bot_GoodsShelves\data\test_out\mian.jpg'
img = cv2.imread(imgfile) # Read in your image
# cv2.imshow('original',img)
img1=img
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# 第一个是原图像(必须为灰度图)，第二个是阈值，第三个叫maxVal，
# 它表示如果这个点的灰度大于(有时是小于，根据第四个输入的设置)阈值，则将这个点的灰度修改成maxVal，第四个参数则决定函数的具体功能
ret, binary = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
x=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy,index =x[1],x[0],x[2]

# The index of the contour that surrounds your object
idx =x[2]
# Create mask where white is what we want, black     otherwise
mask = np.zeros_like(img)
# Draw filled contour in mask，第四个参数：轮廓内的色调；最后一个参数：是否填满
cont_copy =[]
crop_point =[]
m = 0
maxsize =1
for j in range(len(contours)):
    if len(contours[j])>100:
        cont_copy.append(contours[j])
        if len(contours[j])>maxsize:
            maxsize = len(contours[j])
            crop_point = []
            for pp in contours[j]:
                point_tmp = point(pp[0])
                crop_point.append(point_tmp.to_pint())

        print contours[j]
        # im = Image.open(imgfile).convert("RGBA")
        # # convert to numpy (for convenience)
        # imArray = np.asarray(im)
        # # create mask
        # maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
        # ImageDraw.Draw(maskIm).polygon(contours[j], outline=1, fill=1)
        # maskIm.show()
        m+=1

im = Image.open(imgfile).convert("RGBA")
# convert to numpy (for convenience)
imArray = np.asarray(im)
# create mask
maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), (0,0,0))
ImageDraw.Draw(maskIm).polygon(crop_point, outline=20, fill = (90, 190, 0))
maskIm.show()

print ' final size is %d'%m
cv2.drawContours(mask, cont_copy,-1,(160,255,255),-1)
cv2.imshow('mask1',mask)

processed_crop_point = point_process(crop_point,100)

drawPoly(imgfile,processed_crop_point,r'E:\bot_GoodsShelves\data\test_out\img_crop.jpg')
# Extract out the object and place into output image，设置底色为黑色。
out = np.zeros_like(img)
out[mask == 255] = img[mask == 255]
# Show the output image
cv2.imshow('Output', out)
cv2.imwrite('mask1.jpg',out)
cv2.waitKey(0)
cv2.destroyAllWindows()

