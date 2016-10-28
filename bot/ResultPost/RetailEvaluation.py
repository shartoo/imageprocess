# -*- coding:utf8 -*-
'''
功能：测试向领犀发送json文件   货架物品评估  并打印返回內容

本代码使用样例：
  linux：  python RetailEvaluation.py  /home/data/RetailEvaluation.json

其中 /home/data/RetailEvaluation.json 为选手提交的json结果完整路径（注意：json文件必須是無BOM的UTF格式）

'''
import json
import requests
import sys

if __name__=='__main__':
    '''
        @:param 参数为需要的上传到該url的json文件（注意：json文件必须是BOM的UTF格式）
    '''
    if(len(sys.argv)!=1):
        print "usage :<url>  <jsonfile>"
    #url =sys.argv[1]
    url = r"http://www.lingxi.ai:8000/api/RetailEvaluation?teamid=1001"
    filePath =sys.argv[1]
    #filePath ="e:\\test.json"
    headers = {"Content-Type": "application/json"}
    f = file(filePath)
    jsonobj = json.load(f)
    mynewjson=json.dumps(jsonobj)
    print 'print request result...'
    r = requests.post(url, data=mynewjson, headers=headers)
    f.close()
    print r.content
