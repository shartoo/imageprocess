# -*- coding:utf-8 -*-
'''
测试使用python opencv操作图像
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import sys

def corner_detect(pic):
    '''
      边角检测
    :param pic:
    :return:
    '''
    img = cv2.imread(pic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 29, 0.04)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    while (True):
        cv2.imshow('corners', img)
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()

def feature_match(source,target):
    '''
        从一幅图像中匹配另外一幅图像中的特征
    :param source:
    :param target:
    :return:
    '''

    cv2.ocl.setUseOpenCL(False)   # 关键代码，如果不添加，则会导致出错
    img1 = cv2.imread(source,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    kp1,des1 = orb.detectAndCompute(img1,None)
    kp2,des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches= bf.match(des1,des2)
    matches = sorted(matches,key = lambda x:x.distance)
    print kp1
    print "+++++++++++++++++++++++++++++"
    print kp2
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:40],img2,flags=2)
    plt.imshow(img3)
    plt.show()


def  knn_match(source,target):
    cv2.ocl.setUseOpenCL(False)  # 关键代码，如果不添加，则会导致出错
    img1 = cv2.imread(source, 0)
    img2 = cv2.imread(target, 0)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(des1, des2, k=2)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)
    plt.imshow(img3)
    plt.show()


def flann_match(source,target):
    cv2.ocl.setUseOpenCL(False)  # 关键代码，如果不添加，则会导致出错
    queryImage = cv2.imread(source, 0)
    trainingImage = cv2.imread(target, 0)
    # create SIFT and detect/compute
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(queryImage, None)
    kp2, des2 = sift.detectAndCompute(trainingImage, None)
    # FLANN matcher parameters
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)
    # prepare an empty mask to draw good matches
    matchesMask = [[0, 0] for i in xrange(len(matches))]
    # David G. Lowe's ratio test, populate the mask
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    drawParams = dict(matchColor=(0, 255, 0),
                      singlePointColor=(255, 0, 0),
                      matchesMask=matchesMask,
                      flags=0)
    resultImage =cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)
    plt.imshow(resultImage, )
    plt.show()

def cv_draw():
    img = np.zeros((512,512,3),np.uint8)  # 生成一个  空  彩色图像
    cv2.line(img,(0,0),(511,511),(155,155,155),5)
    plt.imshow(img,'brg')
    plt.show()

    # 在一幅图像中矩阵

def draw_rectangle_in_img(img,start,end,color,thin):
    '''
        在一幅图像中化矩阵
    :param img:       需要画矩阵的图像
    :param start:     矩阵的左上角坐标。例如 (100,300)
    :param end:       矩阵的右下角坐标。例如 (800,600)
    :param color:     矩阵颜色，如果是三色通道就是。(0, 255, 0)
    :param thin:      矩阵的笔触大小。是整数 ，例如 4
    :return:
    '''
    cv2.rectangle(im, start, end, color, thin)
    plt.imshow(im,'brg')
    plt.show()



def detect_traffic(datapath):

    def path(cls, i):
        return "%s/%s%d.pgm" % (datapath, cls, i + 1)

    pos, neg = "pos-", "neg-"
    detect = cv2.xfeatures2d.SIFT_create()
    extract = cv2.xfeatures2d.SIFT_create()
    flann_params = dict(algorithm=1, trees=5)
    flann =cv2.FlannBasedMatcher(flann_params, {})
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
    extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)

    def extract_sift(fn):
        im = cv2.imread(fn, 0)
        return extract.compute(im, detect.detect(im))[1]

    for i in range(8):
        bow_kmeans_trainer.add(extract_sift(path(pos, i)))
        bow_kmeans_trainer.add(extract_sift(path(neg, i)))
    voc = bow_kmeans_trainer.cluster()
    extract_bow.setVocabulary(voc)

    def bow_features(fn):
        im = cv2.imread(fn, 0)
        return extract_bow.compute(im, detect.detect(im))

    traindata, trainlabels = [], []
    for i in range(20):
        traindata.extend(bow_features(path(pos, i)))
        trainlabels.append(1)
        traindata.extend(bow_features(path(neg, i)));
        trainlabels.append(-1)

    svm = cv2.ml.SVM_create()
    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

    def predict(fn):
        f = bow_features(fn);
        p = svm.predict(f)
        print fn, "\t", p[1][0][0]
        return p

    car, notcar = "/home/d3athmast3r/dev/python/study/images/car.jpg", \
    "/home/d3athmast3r/dev/python/study/images/bb.jpg"
    car_img = cv2.imread(car)
    notcar_img = cv2.imread(notcar)
    car_predict = predict(car)
    not_car_predict = predict(notcar)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if (car_predict[1][0][0] == 1.0):
        cv2.putText(car_img, 'Car Detected', (10, 30), font, 1,(0, 255, 0), 2, cv2.LINE_AA)
    if (not_car_predict[1][0][0] == -1.0):
        cv2.putText(notcar_img, 'Car Not Detected', (10, 30), font, 1, (0, 0,255), 2, cv2.LINE_AA)

    cv2.imshow('BOW + SVM Success', car_img)
    cv2.imshow('BOW + SVM Failure', notcar_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':

    source="E:\\bot\\0.jpg"
    im = cv2.imread(source)
    target="E:\\bot\\0.jpg"
    #feature_match(source,target)
    cv2.rectangle(im,(100,300),(800,600), (0, 255, 0),1)
    #plt.imshow( im,'brg')
    #plt.show()
    cv_draw()
