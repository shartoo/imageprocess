# -*- coding:utf-8 -*-
'''
测试代码
'''

from PIL import Image
from numpy import *
from imtool import histeq

im = array(Image.open("E:\\bot\\00\\000ac57e2c5211e680ff00505681e231.jpg").convert('L'))
im2,cdf =histeq(im)
im2.show()