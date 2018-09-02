# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:23:16 2018

@author: hp
"""

#####################六、模型效果评估####################
#############################################################
#主要有两个api来实现 CountVectorizer 和 TfidfVectorizer
#CountVectorizer：
#   只考虑词汇在文本中出现的频率
#TfidfVectorizer：
#   除了考量某词汇在文本出现的频率，还关注包含这个词汇的所有文本的数量
#    能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征


import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD  #降维
from sklearn.naive_bayes import BernoulliNB     #伯努利分布的贝叶斯公式
from sklearn.metrics import f1_score,precision_score,recall_score
 
## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
 
#1、文件数据读取
df = pd.read_csv("../data/result_process02",encoding="utf-8",sep=",")
#如果有nan值，进行上删除操作
df.dropna(axis=0,how="any",inplace=True)    #删除表中含有任何NaN的行
print(df.head())
print(df.info())
 
#2、数据分割
x_train,x_test,y_train,y_test = train_test_split(df[["has_date","jieba_cut_content","content_sema"]],
                                                 df["label"],test_size=0.2,random_state=0)
print("训练数据集大小:%d" %x_train.shape[0])
print("测试数据集大小:%d" %x_test.shape[0])
print(x_train.head())
 
#3、开始模型训练
#3.1、特征工程，将文本数据转换为数值型数据
transformer = TfidfVectorizer(norm="l2",use_idf=True)
svd = TruncatedSVD(n_components=20)     #奇异值分解，降维
jieba_cut_content = list(x_train["jieba_cut_content"].astype("str"))
transformer_model = transformer.fit(jieba_cut_content)
df1 = transformer_model.transform(jieba_cut_content)
svd_model = svd.fit(df1)
df2 = svd_model.transform(df1)
 
data = pd.DataFrame(df2)
print(data.head())
print(data.info())
 
#3.2、数据合并
data["has_date"] = list(x_train["has_date"])
data["content_sema"] = list(x_train["content_sema"])
print("========数据合并后的data信息========")
print(data.head())
print(data.info())
 
t1 = time.time()
nb = BernoulliNB(alpha=1.0,binarize=0.0005) #贝叶斯分类模型构建
model = nb.fit(data,y_train)
t = time.time()-t1
print("贝叶斯模型构建时间为:%.5f ms" %(t*1000))
 
#4.1 对测试数据进行转换
jieba_cut_content_test = list(x_test["jieba_cut_content"].astype("str"))
data_test = pd.DataFrame(svd_model.transform(transformer_model.transform(jieba_cut_content_test)))
data_test["has_date"] = list(x_test["has_date"])
data_test["content_sema"] = list(x_test["content_sema"])
print(data_test.head())
print(data_test.info())
 
#4.2 对测试数据进行测试
y_predict = model.predict(data_test)
 
#5、效果评估
print("准确率为:%.5f" % precision_score(y_test,y_predict))
print("召回率为:%.5f" % recall_score(y_test,y_predict))
print("F1值为:%.5f" % f1_score(y_test,y_predict))

