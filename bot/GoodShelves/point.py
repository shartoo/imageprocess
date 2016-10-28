# -*- coding:utf-8 -*-
'''
     图像中像素点
'''

class point:
    def __init__(self,list):
        self.x = list[0]
        self.y = list[1]

    def to_pint(self):
        return (self.x,self.y)