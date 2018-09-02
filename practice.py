# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:22:47 2018

@author: hp
"""
#特征提取练习
import pandas as pd
import numpy as np
import scipy.stats as ss
df=pd.DataFrame({"A":ss.norm.rvs(size=10),"B":ss.norm.rvs(size=10),"C":ss.norm.rvs(size=10),"D":np.random.randint(low=0,high=2,size=10)})

df
#引入svr，决策树回归器
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
#特征标注
X=df.loc[:,["A","B","C"]]
Y=df.loc[:,"D"]

#特征选择的主要方法，有专门的一个包sklearn.feature_selection
#SelectionKBest,RFE,SelectFromModel 过滤式，包裹式，嵌入式特征选择方法
from sklearn.feature_selection import SelectKBest,RFE,SelectFromModel
skb=SelectKBest(k=2)
#拟合 过滤式 bc
skb.fit(X,Y)
skb.transform(X)
#包裹式AC
rfe=RFE(estimator=SVR(kernel="linear"),n_features_to_select=2,step=1)
rfe.fit_transform(X,Y)
#嵌入式b
sfm=SelectFromModel(estimator=DecisionTreeRegressor(),threshold=0.5)
sfm.fit_transform(X,Y)

#特征变换
#特征离散化  等深分箱 等宽分箱
lst=[6,8,10,11,16,17,25,29,31]
pd.qcut(lst,q=3,labels=["low","medium","high"])
pd.cut(lst,bins=3,labels=["low","medium","high"])

#归一化标准化
from sklearn.preprocessing import MinMaxScaler,StandardScaler
MinMaxScaler().fit_transform(np.array([1,4,10,15,21]).reshape(-1,1))
StandardScaler().fit_transform(np.array([1,1,1,1,0,0,0,0]).reshape(-1,1))

#数值化 标签编码 +独热编码
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LabelEncoder().fit_transform(np.array(["Down","medium","medium","up","up","Down"]).reshape(-1,1))

#one-hot
lb_encoder=LabelEncoder()
lb_encoder.fit_transform(np.array(["blue","gree","pink","yellow","red"]))
oht_encoder=OneHotEncoder().fit(lb_tran_f.reshape(-1,1))
oht_encoder.transform(lb_encoder.transform(np.array(["blue","gree","pink","yellow","red"])).reshape(-1,1))
#正规化l1
from sklearn.preprocessing import Normalizer
Normalizer(norm='l1').fit_transform(np.array([1,1,3,-1,2]).reshape(-1,1))
Normalizer(norm='l2').fit_transform(np.array([1,1,3,-1,2]).reshape(-1,1))
Normalizer(norm='l1').fit_transform(np.array([[1,1,3,-1,2]]))
Normalizer(norm='l2').fit_transform(np.array([[1,1,3,-1,2]]))


###房价表的特征预处理
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def h_preprocessing(sl=False,le=False,lo=False,la=False,to=False,po=False,ho=False,me=False):
    df1=pd.read_csv('G:\machine python\housing.csv')
    #1.：得到标注
    label=df1["total_bedrooms"]
    df1=df1.drop("total_bedrooms",axis=1)
    #2.清洗数据 
    df1=df1.dropna(subset=["longitude","latitude"])
    df1=df1[df1["latitude"]<=37.84][df1["housing_median_age"]<=40]
    #3\特征选择
    #4、特征处理
    scaler_lst=[sl,le,lo,la,to,po,ho,me]
    column_lst=["median_income","housing_median_age","longitude","latitude","total_rooms","population","households","median_house_value"]
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i]=="ocean_proximity":
                df1[column_lst[i]]=[map_ocean_proximity(s) for s in df1["ocean_proximity"].values]
            else:
                df1[column_lst[i]]=LabelEncoder().fit_transform(df1[column_lst[i]])
            
            df1[column_lst[i]]=\
            MinMaxScaler().fit_transform(df1[column_lst[i]].values.reshape(1,-1)[0])
        else:
            df1[column_lst[i]]=\
            StandardScaler().fit_transform(df1[column_lst[i]].values.reshape(1,-1)[0])
            
    return df1
d=dict([("NEAR BAY ",0),("<1H OCEAN",1)])
def map_salary(s):
    return d.get(s,0)
        
def main():
    print(h_preprocessing(sl=True,le=True,lo=True,la=True,to=True,po=True,ho=True,me=True))












