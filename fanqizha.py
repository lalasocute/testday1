# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 08:59:53 2018

@author: hp
"""

import pandas as pd
import numpy as np
import sys
#skiprows=1删除第一行,low_memory=True将导入的数据按其格式存储，弊端可能会报错
df = pd.read_csv("G:/LoanStats3a.csv",skiprows=1,low_memory=True)
#只打印前10行
#print(df.head(10))
#信息全部打印、查看数据特征表格信息
#print(df.info())
#删除一个列,
df.drop('id',axis=1,inplace=True)
df.drop('member_id',axis=1,inplace=True)
#删除肉眼看得见的空值,正则匹配term列 将months删除regex=True
df.term.replace(to_replace='[^0-9]+',value='',inplace=True,regex=True)
#利息部分int_rate,df.int_rate.replace('%',value-'',inplace,regex=True) 百分号没有去掉，说明它就是数值型的可以直接参与运算
df.int_rate.replace('%',value='',inplace = True)
print(df.head(5))
#...清洗数据，去除特征中的特征字符...
df.drop('sub_grade',axis = 1,inplace = True)
df.drop('emp_title',axis = 1,inplace = True)

#列出工作年限分段信息
print(df.emp_length.value_counts())
#正则匹配工作年限,将小于1的取为1
########################################################
df.emp_length.replace('n/a',np.nan ,inplace = True)
df.emp_length.replace(to_replace = '[^0-9]+', value = '', inplace = True ,regex = True)

#how ，any一列有一个空值就删除，all全部是空值才删除,axis = 1列，axis = 0 行
df.dropna(axis = 1,how='all',inplace = True)
df.dropna(axis = 0,how='all',inplace = True)
print(df.info())

#删除空值比较多的列
#将空值太多的删除debt_settlement_flag_date     160 non-null object
#debt_settlement_flag_date     160 non-null object
#settlement_status             160 non-null object
#settlement_date               160 non-null object
#settlement_amount             160 non-null float64
#settlement_percentage         160 non-null float64
#settlement_term               160 non-null float64
#样本的观测经验：一般来说，样本的正例和反例个数比为10:1比较好
df.drop(['debt_settlement_flag_date','settlement_status','settlement_date',\
         'settlement_amount','settlement_percentage',\
         'settlement_term'],axis = 1, inplace = True)
#删除不为空，但是特征重复太多的，eg：全是0，全是1，或者全是individual,或者全是n。。。
#删除方法：先删除float类型的列，再删除object的列
#统计下重复值信息，各个特征里面的类型信息  删除float中较多重复的列
#for col in df.select_dtypes(include = ['float']).columns:
   # print('col{} has {}'.format(col,len(df[col].unique())))
#删除特征里面类型过少的列
df.drop(['delinq_2yrs','inq_last_6mths','mths_since_last_delinq','mths_since_last_record',\
         'open_acc','pub_rec','out_prncp','total_acc','out_prncp','out_prncp_inv','collections_12_mths_ex_med',\
         'policy_code','acc_now_delinq','chargeoff_within_12_mths','delinq_amnt','pub_rec_bankruptcies',\
         'tax_liens'],axis = 1, inplace = True)
print(df.info())

    
#删除object类型数据 中较多重复的列
##############################loan_status是标签 ，不能删除
for col in df.select_dtypes(include = ['object']).columns:
    print('col{} has {}'.format(col,len(df[col].unique())))
    
df.drop(['term','grade','emp_length','home_ownership','verification_status',\
         'issue_d','pymnt_plan','purpose','zip_code','addr_state','earliest_cr_line',\
         'initial_list_status','last_pymnt_d','next_pymnt_d','last_credit_pull_d',\
         'application_type','hardship_flag','disbursement_method',\
         'debt_settlement_flag',],axis = 1, inplace = True)


#######检查最后删除信息
#<class 'pandas.core.frame.DataFrame'>
#Int64Index: 42535 entries, 0 to 42535
#Data columns (total 20 columns):
#loan_amnt                  42535 non-null float64
#funded_amnt                42535 non-null float64
#funded_amnt_inv            42535 non-null float64
#int_rate                   42535 non-null object
#installment                42535 non-null float64
#annual_inc                 42531 non-null float64
#loan_status                42535 non-null object
#desc                       29243 non-null object
#title                      42523 non-null object
#dti                        42535 non-null float64
#revol_bal                  42535 non-null float64
#revol_util                 42445 non-null object
#total_pymnt                42535 non-null float64
#total_pymnt_inv            42535 non-null float64
#total_rec_prncp            42535 non-null float64
#total_rec_int              42535 non-null float64
#total_rec_late_fee         42535 non-null float64
#recoveries                 42535 non-null float64
#collection_recovery_fee    42535 non-null float64
#last_pymnt_amnt            42535 non-null float64
#dtypes: float64(15), object(5)
#memory usage: 6.8+ MB
    

#特征的状态数print(df.emp_length.value_counts())

#再删除两列不重要的特征
df.drop(['desc','title'],axis=1,inplace=True)

print(df.loan_status.value_counts())
#以下两个为无法确定的状态，设置为np.nan ，剔除没有标签的样本。。。。，Fully Paid设置为1，Charged Off设置为0
###Does not meet the credit policy. Status:Fully Paid      1988
##Does not meet the credit policy. Status:Charged Off      761
df.loan_status.replace('Fully Paid',value = int(1),inplace = True)
df.loan_status.replace('Charged Off',value = int(0),inplace = True)
df.loan_status.replace('Does not meet the credit policy. Status:Fully Paid',\
                       np.nan, inplace = True)
df.loan_status.replace('Does not meet the credit policy. Status:Charged Off',\
                       np.nan, inplace = True)

#查看样本特征清洗后的数据print(df.info())
##标签二值化
#####删除标签为空值的实例后的样本信息 loan_status   39786 non-null float64
###删除了3000多个实例
df.dropna(subset = ['loan_status'], how = 'any', inplace = True)
print(df.info())

#把样本中的空值用0.0 去填充
df.fillna(0.0,inplace = True)


#================================以上部分为数据清洗================================

##=====================相关性分析===================================
#协方差矩阵  计算清洁后样本特征的相关性，去除多重相关特征（保留1列）eg:有3列特征相似，大致相同，去除两列，保留1列
cor = df.corr()
cor.iloc[:,:] = np.tril(cor, k = -1)
cor = cor.stack()
print(cor[(cor>0.55)|(cor<-0.55)])
####将相关性值大于0.95以上的都删除

df.drop(['loan_amnt','funded_amnt','total_pymnt'],axis = 1, inplace = True)

###########删除线性相关系数>0.95的列
############################哑变量的处理
df = pd.get_dummies(df)

###将处理好的特征存到'G:/feature001.csv'
df.to_csv('G:/feature001.csv')

#print(df.info())

########################=================开始模型训练======================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
###############metrics监督学习的评判标准
from sklearn import metrics
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import time
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
###读取数据
df = pd.read_csv('G:/feature001.csv')

Y = df.loan_status
X = df.drop('loan_status', 1, inplace = False)

x_train,x_test,y_train,y_test = train_test_split(X , Y, test_size = 0.3,random_state = 0)

#==================逻辑回归========================
lr = LogisticRegression()
start = time.time()
lr.fit(x_train,y_train)
train_predict = lr.predict(x_train)
train_f1 = metrics.f1_score(train_predict,y_train)
train_acc = metrics.accuracy_score(train_predict,y_train)
train_rec = metrics.recall_score(train_predict,y_train)
print("逻辑回归模型上的效果如下： ")
print("在训练集上f1_mean的值为%.4f"%train_f1,end=' ')
print("在训练集上精确值为%.4f"%train_acc,end=' ')
print("在训练集上查全率值为%.4f"%train_rec)
#===========测试集上
test_predict = lr.predict(x_test)
test_f1 = metrics.f1_score(test_predict,y_test)
test_acc = metrics.accuracy_score(test_predict,y_test)
test_rec = metrics.recall_score(test_predict,y_test)

print("在测试集上f1_mean的值为%.4f"%test_f1,end=' ')
print("在测试集上精确值为%.4f"%test_acc,end=' ')
print("在测试集上查全率值为%.4f"%test_rec)
###====测试时间
end = time.time()
print(end-start)


#=====================随机森林=======================
print("随机森林的效果如下"+"="*30)
rf = RandomForestClassifier()
start = time.time()
rf.fit(x_train,y_train)
train_predict = rf.predict(x_train)
train_f1 = metrics.f1_score(train_predict,y_train)
train_acc = metrics.accuracy_score(train_predict,y_train)
train_rec = metrics.recall_score(train_predict,y_train)
print("随机森林模型上的效果如下： ")
print("在训练集上f1_mean的值为%.4f"%train_f1,end=' ')
print("在训练集上精确值为%.4f"%train_acc,end=' ')
print("在训练集上查全率值为%.4f"%train_rec)
#===========测试集上
test_predict = rf.predict(x_test)
test_f1 = metrics.f1_score(test_predict,y_test)
test_acc = metrics.accuracy_score(test_predict,y_test)
test_rec = metrics.recall_score(test_predict,y_test)

print("在测试集上f1_mean的值为%.4f"%test_f1,end=' ')
print("在测试集上精确值为%.4f"%test_acc,end=' ')
print("在测试集上查全率值为%.4f"%test_rec)
###====测试时间
end = time.time()
print(end-start)

#================================svm=支持向量机========================
#parameters={'kernel':['linear','sigmoid','poly'],'C':[0.01,1],'probability':[True,False]}
#clf = GridSearchCV(svm.SVC(random_state=0),param_grid,cv=5) = parameters
#clf.fit(x_train,y_train)
#print("最优参数是： "，end=' ')
#print(clf.best_params_)
#print('最优模型准确率是： ',end = ' ')
#print(clf.best_score_)
#end = time.time()
#print(end-start)


##==============将重要特征提出来做一个排名======
feature_importance = rf.feature_importances_
feature_importance = 100.0*(feature_importance/feature_importance.max())
index = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10),feature_importance[index],color = 'dodgerblue', alp=0.4)
print(np.array(X.columns)[index])
plt.yticks(np.arange(10+0.25),np.array(X.columnd)[index])
plt.xlabel('Relative importance')
plt.title('Top 10 importance Variable')
plt.show()
























