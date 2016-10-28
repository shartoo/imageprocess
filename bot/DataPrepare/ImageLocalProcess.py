# -*- coding:utf-8 -*-
'''
图像处理局部
'''

from scipy.ndimage import filters
from numpy import *


def compute_harris_reponse(im,sigma=3):
    '''
    在一幅灰度图像中，对每个像素计算Harris角点 检测器相应函数
    :param im:
    :param sigma:
    :return:
    '''
    imx = zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
    imy = zeros(im.shape)
    # 计算Harris矩阵的分量
    Wxx =filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # 计算特征值和迹
    Wdet = Wxx*Wyy-Wxy**2
    Wtr =Wxx+Wyy

    return Wdet/Wtr





