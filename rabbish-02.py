# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 09:25:13 2018

@author: hp
"""

import re
import math
from sqlite3 import dbapi2 as sqlite


# 该函数用于提取特征，将doc中所有不重复的单词提取作为doc的特征
def getwords(doc):
    # 正则表达式，按照非单词字符将句子切分
    splitter = re.compile(r'\W*')
    # 根据非字母字符进行单词切分
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]
    # 返回一组不重复的单词
    return dict([w, 1] for w in words)

# 一次训练多条数据，其参数cl是类 classifier 的一个对象
def sampletrain(cl):
    cl.train('Nobody owns the water.', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('buy pharmaceuticals now', 'bad')
    cl.train('make quick money at the online casino', 'bad')
    cl.train('the quick brown fox jumps', 'good')
    
class classifier:

    def __init__(self, getfeatures, filename=None):
        # 统计特征/分类组合的数量  feature_count简写fc
        self.fc = {}
        # 统计每个分类中的文档数量 category_count简写cc
        self.cc = {}
        # 初始化提取特征的方法函数
        self.getfeatures = getfeatures
     def setdb(self, dbfile):
        self.con = sqlite.connect(dbfile)
        self.con.execute('create table if not exists fc(feature, categories, count)')
        self.con.execute('create table if not exists cc(categories, count)')

     def incf(self, f, cat):
        res = self.con.execute('select count from fc where feature="%s" and categories="%s"' % (f, cat)).fetchone()
        # 如果没有找到这条记录则插入
        if res == None:
            self.con.execute('insert into fc (feature, categories, count) values ("%s", "%s", 1)' % (f, cat))
        else:
            count = float(res[0])
            self.con.execute('update fc set count=%d where feature="%s" and categories="%s"' % (count+1, f, cat))

     def incc(self, cat):
        res = self.con.execute('select count from cc where categories="%s"' % cat).fetchone()
        if res is None:
            self.con.execute('insert into cc (categories, count) values ("%s", 1)' % cat)
        else:
            count = float(res[0])
            self.con.execute('update cc set count=%d where categories="%s"' %(count+1, cat))

     def fcount(self, f, cat):
        res = self.con.execute('select count from fc where feature="%s" and categories="%s"' % (f, cat)).fetchone()
        if res is None:
            return 0
        else:
            return float(res[0])

     def catcount(self, cat):
        res = self.con.execute('select count from cc where categories="%s"' % cat).fetchone()
        if res is None:
            return 0
        else:
            return float(res[0])

     def totalcount(self):
        res = self.con.execute('select count from cc').fetchall()
        return sum(res[i][0] for i in range(len(res)))

     def categories(self):
        res = self.con.execute('select categories from cc').fetchall()
        return [res[i][0] for i in range(len(res))]

    # train函数接受一条训练数据，首先提取特征，随后维护self.cc和self.fc两个表
     def train(self, item, cat):
        features = self.getfeatures(item)
        # 维护表fc, 针对该分类为每个特征增加计数值
        for f in features:
            self.incf(f, cat)
        # 维护表cc
        self.incc(cat)
        self.con.commit()

    # 计算概率，计算P(f|cat)条件概率，即特征f在类别cat条件下出现的概率
     def fprob(self, f, cat):
        # 如果该类别文档数为0则返回0
        if self.catcount(cat) == 0:
            return 0
        return float(self.fcount(f, cat)) / float(self.catcount(cat))

    # 对fprob的条件概率计算方法进行优化，设置初始概率为0.5，权值为1
    # 参数： f为特征, cat为类别，prf为self.prob，weight为初始值ap所占权重
     def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
         # 计算当前条件概率
        basicprob = prf(f, cat)
        # 统计特征在所有分类中出现的次数
        totals = sum([self.fcount(f, c) for c in self.categories()])
        # 计算加权平均
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp



class naivebayes(classifier):
    # 初始化阈值列表为空
    def __init__(self, getfeatures):
        classifier.__init__(self, getfeatures)
        self.thresholds = {}

    # 计算整片文章的属于cat类的概率  等于文章中所有单词属于cat类的条件概率之积
    def docprob(self, item, cat):
        features = self.getfeatures(item)
        # 将所有特征的概率相乘
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p

    # P(cat|item) = P(item|cat) * P(cat) / P(item) 其中p(item|cat)通过上一个函数计算得到
    # 其中 P(item) 这一项由于不参与比较，因此可以忽略
    def prob(self, item, cat):
        catprob = float(self.catcount(cat)) / float(self.totalcount())
        docprob = self.docprob(item, cat) * catprob
        return docprob

    # 设置阈值
    def setthresholds(self, cat, t):
        self.thresholds.setdefault(cat, 1)
        self.thresholds[cat] = t

    # 获取阈值
    def getthresholds(self, cat):
        if cat in self.thresholds.keys():
            return self.thresholds[cat]
        return 1.0

    # 根据prob计算出的所有cat类的P(cat|item)进行比较，同时根据设定的thresholds阈值，将item判定到某一类
    def classify(self, item, default=None):
        # 构建(类别，概率)列表，并按照概率排序
        prob = sorted([(cat, self.prob(item, cat)) for cat in self.categories()], key=lambda x: x[1], reverse=True)
        print prob[:]

        # 如果 最大概率 > 阈值 * 次大概率，则判断为最大概率所属类别，否则判定为default类别
        if prob[0][1] > self.getthresholds(prob[0][0]) * prob[1][1]:
            return prob[0][0]
        else:
            return default
        
        
        
class fisherclassifier(classifier):
    # 设定分类概率的下限，如将bad的分类下线设置为0.6，则只有当P(bad|item)>0.6时才判定为bad
    def __init__(self, getfeatures):
        classifier.__init__(self, getfeatures)
        self.minimums = {}

    # 计算条件概率 P(cat|f)
    def cprob(self, f, cat):
        # P(f|cat)
        clf = self.fprob(f, cat)
        if clf == 0:
            return 0

        # P(f): 特征在所有分类中出现的频率
        freqsum = sum([self.fprob(f, c) for c in self.categories()])

        # 概率等于特征在该分类中出现的频率除以总体概率
        p = clf / freqsum
        return float(p)

    # 求一篇文档item的概率
    # 将各特征的概率值组合起来，连乘，然后去自然对数，再讲所得结果乘以-2，最后利用倒置对数卡方函数求得概率
    def fisherprob(self, item, cat):
        # 将所有概率值相乘
        p = 1.0
        features = self.getfeatures(item)
        for f in features:
            # 加入初始权值，计算概率
            p *= (self.weightedprob(f, cat, self.cprob))
            #print p

        # 取自然对数，乘-2
        fscore = -2.0 * math.log(p)

        # 对数卡方函数
        return self.invchi2(fscore, len(features) * 2)

    # ??? 不懂
    def invchi2(self, chi, df):
        m = chi / 2.0
        sum1 = math.exp(-m)
        term = sum1
        for i in range(1, df//2):
            term *= m / float(i)
            sum1 += term
        return min(sum1, 1.0)

    # 设定分类概率的下限，如将bad的分类下线设置为0.6，则只有当P(bad|item)>0.6时才判定为bad
    def setminimum(self, cat, min):
        self.minimums[cat] = min

    def getminimum(self, cat):
        if cat not in self.minimums:
            return 0
        return self.minimums[cat]

    # 分类
    def classify(self, item, default=None):
        # 循环遍历寻找概率值最佳的结果
        best = default
        maxprob = 0.0
        for c in self.categories():
            p = self.fisherprob(item, c)
            # 确保其超过下限
            if p > self.getminimum(c) and p > maxprob:
                best = c
                maxprob = p
            print '(', c, ', ', p, ')'
        return best




if __name__ == '__main__':
    # 测试 getwords 函数
    """
    doc = "I am a teacher and he is a student"
    doc_words = getwords(doc)
    print doc_words.keys()
    """

    # 测试训练函数train
    """
    cl = classifier(getwords)
    cl.train('the quick brown fox jumps over the lazy dog', 'good')
    cl.train('make quick money in the online casino', 'bad')
    print cl.fcount('quick', 'good')
    print cl.fcount('quick', 'bad')
    """

    # 测试计算条件概率
    """
    cl = classifier(getwords)
    sampletrain(cl)
    #print cl.fprob('money', 'good')
    print cl.weightedprob('money', 'good', cl.fprob)
    sampletrain(cl)
    print cl.weightedprob('money', 'good', cl.fprob)
    """

    # 测试朴素贝叶斯计算文档概率
    """
    cl = naivebayes(getwords)
    sampletrain(cl)
    print cl.prob('quick rabbit', 'good')
    print cl.prob('quick rabbit', 'bad')
    """

    # 测试朴素贝叶斯分类器依照不同的阈值判断类别
    """
    cl = naivebayes(getwords)
    sampletrain(cl)
    print cl.classify('quick money')
    # 设置P(bad|item) > 3.0 * P(good|item)时才判定为bad
    cl.setthresholds('bad', 3.0)
    print cl.classify('quick money')

    for i in range(10):
        sampletrain(cl)
    print cl.classify('quick money')
    """

    # 测试基于fisher方法的分类器
    """
    cl = fisherclassifier(getwords)
    sampletrain(cl)
    #print cl.cprob('quick', 'good') + cl.cprob('quick', 'bad')
    print cl.cprob('money', 'bad')

    # 整片文档的概率
    print cl.fisherprob('quick rabbit', 'good')
    print cl.fisherprob('quick rabbit', 'bad')

    # 整片文档的类别
    cl.setminimum('bad', 0.8)
    cl.setminimum('good', 0.4)
    print cl.classify('quick money')
    """
    # 测试改写后的操作sqlite数据库的函数是否正确
    cl = fisherclassifier(getwords)
    cl.setdb('test.db')
    #sampletrain(cl)
    res1 = cl.con.execute('select * from fc').fetchall()
    res2 = cl.con.execute('select * from cc').fetchall()
    res3 = cl.catcount('good')
    #print res3

    # 构建朴素贝叶斯分类器，直接使用已经建立好的数据库
    cl2 = naivebayes(getwords)
    cl2.setdb('test.db')
    res4 = cl2.classify('quick money')
    print res4
        
