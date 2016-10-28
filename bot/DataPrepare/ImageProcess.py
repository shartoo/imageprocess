# -*- coding:utf-8 -*-
'''
处理图像，旋转，白化，扭曲等操作
'''
from PIL import Image
from PIL import ImageDraw
import random
import sys

def spacFilter(mode,im):
    draw = ImageDraw.Draw(im)
    for i in range(0,list(im.size)[0]):
        for j in range(0,list(im.size)[1]):
            if ((i - 1) >= 0) and ((j - 1) >= 0) and ((i + 1) < list(im.size)[0]) and ((j + 1) < list(im.size)[1]):
                pixels = [im.getpixel((i + 1,j + 1)),im.getpixel((i + 1,j)),im.getpixel((i + 1,j - 1)), \
                      im.getpixel((i,j + 1)),im.getpixel((i,j - 1)), \
                      im.getpixel((i - 1,j + 1)),im.getpixel((i - 1,j)),im.getpixel((i - 1,j - 1))]
            else:
                continue
            if mode == "mean":       #3x3平均值滤波
                #print pixels
                #color = 128
                datax = list(pixels)
                color = (datax[0] + datax[1] + datax[2] + datax[3] + datax[4] + datax[5] + \
                         datax[6] + datax[7]) / 8
            elif mode == "median":  #3x3中值滤波
                color = (pixels[0] + pixels[1] + pixels[2] + pixels[3] + pixels[4] + pixels[5] + \
                         pixels[6] + pixels[7]) / 8
            elif mode == "max":     #3x3最大值滤波
                color = max(pixels)
            elif mode == "min":     #3x3最小值滤波
                color = min(pixels)
            point = [i,j]
            draw.point(point,color)
    del draw
    return im


def addNoise(im,mode,value):
    '''
       添加噪声
    :param im:
    :param mode:
    :param value:
    :return:
    '''
    draw = ImageDraw.Draw(im)
    for i in range(0,list(im.size)[0]):
        for j in range(0,list(im.size)[1]):
            if mode == "uniform":
                rnd = random.uniform(value[0],value[1])
            elif mode == "normal":                         #UNIFORM噪声
                rnd = random.gauss(value[0],value[1])
            elif mode == "lognormal":
                rnd = random.lognormvariate(value[0],value[1])
            elif mode == "negexp":
                rnd = random.expovariate(value[0],value[1])
            elif mode == "gamma":
                rnd = random.gammavariate(value[0],value[1])
            elif mode == "beta":
                rnd = random.betavariate(value[0],value[1])
            elif mode == "pareto":
                rnd = random.paretovariate(value[0])
            elif mode == "weibull":
                rnd = random.weibullvariate(value[0],value[1])

            if im.mode == "RGB":
                color = list(im.getpixel((i,j)))
                color[0] = color[0] + rnd
                color[1] = color[1] + rnd
                color[2] = color[2] + rnd
                point = [i,j]
                draw.point(point,tuple(color))
            elif im.mode == "L":
                color = im.getpixel((i,j))
                color = color + rnd
                point = [i,j]
                draw.point(point,color)
            else:
                print "File type not supported!"
                sys.exit(1)
    del draw
    return im

if __name__=='__main__':
    im = Image.open("E:\\bot\\00\\000aadd62bd611e69e0400505681e231.jpg")  # 打开文件[支持大部分图像文件格式]
    #im.show()  # 显示图像
    #im.save("hello.gif", "GIF")  # 保存图像为gif格式
    # 进行基本处理
    im = addNoise(im, "uniform", [0, 50])  # 添加uniform噪声
    im = spacFilter("median", im)  # 中值滤波
    im = im.filter(ImageFilter.EDGE_ENHANCE)  # 基本滤波之 边缘增强
    im = im.convert("L")  # 转换为灰度图像
    im.show()