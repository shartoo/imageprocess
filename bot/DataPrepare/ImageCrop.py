# -*- coding:utf-8 -*-
'''
图像切割
'''
from PIL import Image,ImageDraw
import sys
import numpy as np
import json

def drawPoly(imgfile,polygon):
    # read image as RGB and add alpha (transparency)
    im = Image.open(imgfile).convert("RGBA")
    # convert to numpy (for convenience)
    imArray = np.asarray(im)
    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)
    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape, dtype='uint8')
    # colors (three first columns, RGB)
    newImArray[:, :, :3] = imArray[:, :, :3]
    # transparency (4th column)
    newImArray[:, :, 3] = mask * 255
    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")
    newIm.save("img_crop.png")


def draw_rectangle(imgfile,rectangle,target_img):
    '''
       从图像中切割出一个正方形
    :param imgfile:   需要切割的图像
    :param rectangle: 正方形的左上角和右下角坐标 如 (100,200,400,500)
    :return:
    '''
    # read image as RGB and add alpha (transparency)
    im = Image.open(imgfile).convert("RGBA")
    # convert to numpy (for convenience)
    imArray = np.asarray(im)
    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).rectangle(rectangle, outline=1, fill=(0, 255, 0))
    mask = np.array(maskIm)
    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape, dtype='uint8')
    # colors (three first columns, RGB)
    newImArray[:, :, :3] = imArray[:, :, :3]
    # transparency (4th column)
    newImArray[:, :, 3] = mask * 255
    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")
    newIm.save(target_img)
    print "img %s crop done .."%target_img


'''
        json文件必须为 无BOM的UTF-8格式才能正确解读
'''
def jsonStr2Arr(jsonfile):

    poly_dic ={}
    with open(jsonfile) as f:
        jsonobj = json.load(f)
        for jj in jsonobj:
            categorys = jj['products']



def circle(img):
    ima = Image.open(img).convert("RGBA")
    size = ima.size
    # 因为是要圆形，所以需要正方形的图片
    r2 = min(size[0], size[1])
    if size[0] != size[1]:
        ima = ima.resize((r2, r2), Image.ANTIALIAS)
    imb = Image.new('RGBA', (r2, r2),(255,255,255,0))
    pima = ima.load()
    pimb = imb.load()
    r = float(r2/2) #圆心横坐标
    for i in range(r2):
        for j in range(r2):
            lx = abs(i-r+0.5) #到圆心距离的横坐标
            ly = abs(j-r+0.5)#到圆心距离的纵坐标
            l  = pow(lx,2) + pow(ly,2)
            if l <= pow(r, 2):
                pimb[i,j] = pima[i,j]
    imb.save("test_circle.png")

def circle_new(img,r2):
    ima = Image.open(img).convert("RGBA")
    size = ima.size
    #r2 = min(size[0], size[1])
    #if size[0] != size[1]:
    #    ima = ima.resize((r2, r2), Image.ANTIALIAS)
    ima = ima.resize((r2, r2), Image.ANTIALIAS)
    circle = Image.new('L', (r2, r2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, r2, r2), fill=255)
    alpha = Image.new('L', (r2, r2), 255)
    alpha.paste(circle, (0, 0))
    ima.putalpha(alpha)
    ima.save('test_circle_new.png')

if __name__=='__main__':
    img ='E:\\bot\\deskgoods\\Train\Potato Chips\\0fa7c064bb8d428b8fa1053b56ed35e9.jpg'
    arr =[(547, 13),(494, 61),(524, 169),(589, 198),(733, 198),
          (918, 157),(948, 234),(942, 240),(954, 390),(972, 527),
          (1001, 635),(1025, 713),(1055, 809),(1013, 922),(1013, 958),
          (1205, 946),(1343, 922),(1420, 892),(1492, 898),(1749, 946),
          (1862, 916),(2509, 928),(2562, 892),(3089, 874),(3208, 832),
          (3161, 725),(3107, 605),(3226, 581),(3250, 336),(3256, 192),
          (3256, 109),(3232, 19),(3029, 25),(2957, 49),(2922, 67),(2910, 79),(547, 13)]
    #drawPoly(img,arr)
    jsonfile ='E:\\bot\\Tagging Data\\Tagging Data\\potatoTags.json'
    #jsonStr2Arr(jsonfile)
    draw_rectangle('e:\\bot\\02\\0d6600f82c3e11e69e0400505681e231.jpg',(200,200,600,600),'abc.jpg')