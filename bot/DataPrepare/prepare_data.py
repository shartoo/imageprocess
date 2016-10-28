# -*- coding:utf-8 -*-

# 准备训练集数据

import random
import sys
import os

import MySQLdb

try:
    conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='root',db='bot_second_round',port=3306)
    cur=conn.cursor()
    print cur.execute('select * from base_full')
    cur.close()
    conn.close()
except MySQLdb.Error,e:
     print "Mysql Error %d: %s" % (e.args[0], e.args[1])