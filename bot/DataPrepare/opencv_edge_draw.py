# -*- coding:utf-8 -*-
'''
尝试使用opencv 来做边缘检测
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt #plt.plot(x,y) plt.show()
import os

def detect_edge(img):
    img = cv2.imread(img)
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()

    regions = mser.detectRegions(gray, None)

    for p in regions:
        print p
        print "one array done...."

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    # cv2.putText(vis, str('change'), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))
    # cv2.fillPoly(vis, hulls, (0, 255, 0))


    # cv2.imwrite("test.png", vis)
    cv2.imshow('img', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_cicle(img,targetPath,file):
    image = cv2.imread(img)
    # 创建一个图像副本，作为对照
    output = image.copy()
    # 转换为 单一通道的图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 30)
    # param2参数意义为 累计阈值，此值越大，最终获取的圆形越小，并且是圆的准确率越高
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,20, param1=30,param2=40,minRadius=10,maxRadius=30)

    file = file.replace(".jpg","")
    # ensure at least some circles were found
    print 'begin..'
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        i =0
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # Crop from x, y, w, h -> 100, 200, 300, 400
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            r = r+10   # 原始的圆形 半径太小，此处作为增加范围
            crop_img = image[y-r:y+r,x-r:x+r,:]
            #cv2.imshow("cropped", crop_img)
            saveImg = targetPath+file+"_crop_"+str(i)+".jpg"
            cv2.imwrite(saveImg,crop_img)
            print 'crop image %s saved!...'%saveImg
            cv2.waitKey(0)
            i+=1


        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        cv2.imshow("output", np.hstack([output]))
        cv2.waitKey(0)

if __name__=='__main__':
    path = 'E:\\bot\\traffic_sign_crop\\new\\236\\'
    targetPath = "e:\\bot\\traffic_sign_crop\\236\\"
    allfile = os.listdir(path)
    #for f in allfile:
    #    print "img file is :  %s  " % f
    #    find_cicle(os.path.join(path, f), targetPath, f)


    img = 'e:\\bot\\traffic_sign_crop\\new\\1\\2ad293ee2c5d11e680ff00505681e231.jpg'
    #img ='E:\\207eef1e2c6211e69e0400505681e231_crop_259.jpg'
    #img ='e:\\bot\\02\\0eb3e3d62bb611e680ff00505681e231.jpg'   # min r=5,max r=10
    find_cicle(img,targetPath,'207eef1e2c6211e69e0400505681e231_crop_259.jpg')

