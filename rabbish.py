# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:28:30 2018

@author: hp
"""

import jieba
#posseg词性标注
import jieba.posseg as pseg
jieba.load_userdict('./word.txt')

str = '今天天气不错，我们很开心来到长安大学上大数据的课程'
#用“/”分词
cut1 = jieba.cut(str)#一般用这个
cut2 = jieba.cut(str,cut_all = True)#全模式
cut3 = jieba.cut(str,HMM = False)
print('/'.join(cut1))
print('/'.join(cut2))
print('/'.join(cut3))

cut4 = pseg.cut(str)
for w in cut4:
    print(w.word,end='')
    #a标注词性
    print(w.flag)
    
    ##############
###########数据处理############
