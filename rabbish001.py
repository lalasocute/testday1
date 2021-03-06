# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:23:15 2018

@author: hp
"""

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:ZhengzhengLiu
 
import os
 
#1、索引文件(分类标签)读取，该文件中分为两列
#第一列：分类标签是否为垃圾邮件（是：spam、否：ham）；
# 第二列：存放邮件对应文件夹路径，两列之间通过空格分割

def read_index_file(file_path):
    type_dict = {"spam":"1","ham":"0"}      #用字典存放垃圾邮件的分类标签
    index_file = open(file_path)
    index_dict = {}
    try:
        for line in index_file:  # 按行循环读取文件
            arr = line.split(" ")  # 用“空格”进行分割
            #pd.read_csv("full/index",sep=" ")      #pandas来写与上面等价
            if len(arr) == 2:       #分割完之后如果长度是2
                key,value = arr     ##分别将spam  ../data/178/129赋值给key与value
            #添加到字段中
            value = value.replace("../data","").replace("\n","")    #替换
            # 字典赋值，字典名[键]=值，lower()将所有的字母转换成小写
            index_dict[value] = type_dict[key.lower()]      #
    finally:
        index_file.close()
    return index_dict
 
#2、邮件的文件内容数据读取
def read_file(file_path):
    # 读操作，邮件数据编码为"gb2312",数据读取有异常就ignore忽略
    file = open(file_path,"r",encoding="gb2312",errors="ignore")
    content_dict = {}
 
    try:
        is_content = False
        for line in file:  # 按行读取
            line = line.strip()  # 每行的空格去掉用strip()
            if line.startswith("From:"):
                content_dict["from"] = line[5:]
            elif line.startswith("To:"):
                content_dict["to"] = line[3:]
            elif line.startswith("Date:"):
                content_dict["data"] = line[5:]
            elif not line:
                # 邮件内容与上面信息存在着第一个空行，遇到空行时，这里标记为True以便进行下面的邮件内容处理
                # line文件的行为空时是False，不为空时是True
                is_content = True
 
            # 处理邮件内容（处理到为空的行时接着处理邮件的内容）
            if is_content:
                if "content" in content_dict:
                    content_dict["content"] += line
                else:
                    content_dict["content"] = line
    finally:
        file.close()
 
    return content_dict
 
#3、邮件数据处理(内容的拼接,并用逗号进行分割)
def process_file(file_path):
    content_dict = read_file(file_path)
 
    #进行处理(拼接),get()函数返回指定键的值，指定键的值不存在用指定的默认值unkown代替
    result_str = content_dict.get("from","unkown").replace(",","").strip()+","
    result_str += content_dict.get("to","unkown").replace(",","").strip()+","
    result_str += content_dict.get("data","unkown").replace(",","").strip()+","
    result_str += content_dict.get("content","unkown").replace(",","").strip()
    return result_str
 
#4、开始进行数据处理——函数调用

## os.listdir    返回指定的文件夹包含的文件或文件夹包含的名称列表
index_dict = read_index_file(' ')
list0 = os.listdir('G:/rabbish-data/Chinese_Spam_Filter-master/data')      #list0是范围为[000-215]的列表
# print(list0)
for l1 in list0:    # l1:循环000--215
    l1_path ='G:/rabbish-data/Chinese_Spam_Filter-master/data' + l1      #l1_path   ../data/data/215
    print('开始处理文件夹:' + l1_path)
    list1 = os.listdir(l1_path)     #list1:['000', '001', '002', '003'....'299']
    # print(list1)
    write_file_path = 'G:\rabbish-data\Chinese_Spam_Filter-master\data\process01_' + l1
    with open(write_file_path, "w", encoding='utf-8') as writer:
        for l2 in list1:  # l2:循环000--299
            l2_path = l1_path + "/" + l2  # l2_path   ../data/data/215/000
            # 得到具体的文件内容后，进行文件数据的读取
            index_key = "/" + l1 + "/" + l2  # index_key:  /215/000
 
            if index_key in index_dict:
                # 读取数据
                content_str = process_file(l2_path)
                # 添加分类标签（0、1）也用逗号隔开
                content_str += "," + index_dict[index_key] + "\n"
                # 进行数据输出
                writer.writelines(content_str)
 
# 再合并所有第一次构建好的内容
with open('G:\rabbish-data\Chinese_Spam_Filter-master\data\result_process01', 'w', encoding='utf-8') as writer:
    for l1 in list0:
        file_path = 'G:\rabbish-data\Chinese_Spam_Filter-master\data\process01_' + l1
        print("开始合并文件:" + file_path)
 
        with open(file_path, encoding='utf-8') as file:
            for line in file:
                   writer.writelines(line)