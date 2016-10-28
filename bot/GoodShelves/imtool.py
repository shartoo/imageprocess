# -*- coding:utf-8 -*-
'''
  图像处理的工具代码
'''
from PIL import Image,ImageDraw
from numpy import *
import cv2
from matplotlib import pyplot as plt
from skimage.measure import structural_similarity as ssim
import numpy as np
from pylab import *

def histeq(im,bins=256):
    '''
       将一幅灰度图像进行直方图均衡化
    :param im:    灰度图
    :param bins:
    :return:
    '''
    # 计算图像的直方图
    imhist,bins = histogram(im.flatten(),bins,normed=True)
    cdf = imhist.cumsum()  # 累计概率分布函数
    cdf = 255*cdf/cdf[-1]  #归一化

    # 使用累计分布函数的线性差值，计算新的像素值
    im2 = interp(im.flatten(),bins[-1],cdf)
    return im2.reshape(im.shape),cdf

def compute_average(imlist):
    '''
    计算图像列表的平均图像
    :param imlist:
    :return:
    '''

    averageim = array(Image.open(imlist[0]),'f')
    for name in imlist[1:]:
        try:
            averageim +=array(Image.open(name),'f')
        except:
            print name+"....skipped.."

    averageim /=len(imlist)
    return array(averageim,'unit8')

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err



def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()

def template_compare(template,img_file):
    '''
         测试模板比对算法
    :param template:    模板图片
    :param img_file:    匹对的图片
    :return:
    '''
    img = cv2.imread(img_file,0)
    img2 = img.copy()
    template = cv2.imread(template,0)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, 255, 2)

        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()

def drawPoly(imgfile,polygon,target_file):
    '''
        从一幅图像中切割出 polygon指定的坐标点构成的轮廓，并保存到目标图片
    :param imgfile:       需要切割的图片
    :param polygon:       切割时指定的多边形
    :param target_file:    切割之后保存的图片位置
    :return:
    '''
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
    newIm.save(target_file)
    print 'file  %s  crop '%target_file

def point_process(mylist,size):
    '''
        对列表中的坐标点做处理，只保留一个坐标列表中x轴或y轴的 20分位点
    :param mylist:
    :return:
    '''
    resultlist =[]
    minX =0
    maxX =0
    minY =0
    maxY =0
    # 第一次遍历 找到坐标列表中极值的x和y
    for xy in mylist:
        x = xy[0]
        y = xy[1]
        if x>maxX:
            maxX =x
        if x<minX:
            minX =x
        if y > maxY:
            maxY = y
        if y < minY:
            minY = y

    xrange = [minX+(i*(maxX-minX))/size for i in range(size)]
    yrange = [minY + (i * (maxY - minY)) / size for i in range(size)]
    shareX = float((maxX-minX)/float(size))
    shareY = float((maxY - minY)/float(size))

    for xy in mylist:
        x = xy[0]
        y = xy[1]
        x = xrange[int((x-minX)/shareX)-1]
        print (int((y - minY) / shareY)-1)
        x = yrange[int((y - minY) / shareY)-1]
        if  [x,y] not in resultlist:
            resultlist.append([x,y])
    print "before process list size is %d,after is %d"%(len(mylist),len(resultlist))
    return resultlist



def point_list_sort(mylist):
    '''
       对包含数轴点(x,y)的列表排序，先按x，再按y
    :param mylist:
    :return:
    '''


def multi_object_find(tmp,search):
    '''
        多目标检测，从搜索图中匹配模板
    :param tmp:      模板图片
    :param search:   检索的图片
    :return:
    '''
    img_rgb = cv2.imread(search)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(tmp, 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('res.png', img_rgb)

def create_paths():
    '''
       创建文件夹
    :return:
    '''
    import os
    for i in range(88):
        j = 56 + i
        path = 'E:\\bot\\traffic_sign_crop'
        newpath = os.path.join(path, str(j))
        if not os.path.exists(newpath):
            os.mkdir(newpath)
            print 'path %s   created...' % newpath

def test_simlarity():
    # load the images -- the original, the original + contrast,
    # and the original + photoshop
    original = cv2.imread("E:\\imgtest\\1_crop_0.jpg")
    contrast = cv2.imread("E:\\imgtest\\noturn.jpg")
    shopped = cv2.imread("E:\\imgtest\\nopark.jpg")

    res_org = cv2.resize(original, (32, 32))
    res_con = cv2.resize(contrast, (32, 32))
    res_shop = cv2.resize(shopped, (32, 32))

    original = res_org.copy()
    contrast = res_con.copy()
    shopped = res_shop.copy()

    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

    # initialize the figure
    fig = plt.figure("Images")
    images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)

    # loop over the images
    for (i, (name, image)) in enumerate(images):
        # show the image
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis("off")

    # show the figure
    plt.show()

    # compare the images
    compare_images(original, original, "Original vs. Original")
    compare_images(original, contrast, "Original vs. Contrast")
    compare_images(original, shopped, "Original vs. Photoshopped")

if __name__=='__main__':
    temp ='E:\\imgtest\\dagu_temp.jpg'
    search ='E:\\imgtest\\076b390ffbfe4d83b7fcc0b5c4661bdf.jpg'
    #template_compare(temp,search)
    #multi_object_find(temp,search)
    im,cdf = histeq(Image.open(search).convert('L'))
    imshow(im)


