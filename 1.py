# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:24:59 2020

@author: Dell
"""

import numpy
import pandas
import statsmodels.api as sm
import seaborn
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
# Đọc dữ liệu csv
data = pandas.read_csv('LasVegasTripAdvisorReviews-Dataset1.csv', low_memory=False)
# Tinh chỉnh câu hỏi nghiên cứu
sub1=data[(data['Score']>=3) & (data['Score']<=5)]
recode = {"no":0 ,"yes":1}
# Mã hóa biến phân loại theo kiểu nhị phân cho biến Pool
recode1 = {"NO":0 ,"YES":1}
sub1['Pool1']= sub1['Pool'].map(recode1)
# Chuyển sang dữ liệu số
sub1['Pool1'] = pandas.to_numeric(sub1['Pool1'], errors='coerce')
# Canh chuẩn biến Nr. reviews
sub1['Nrreviews_c'] = (sub1['Nr. reviews'] - sub1['Nr. reviews'].mean())
sub2 = sub1[['Nrreviews_c', 'Pool1']].dropna()
log_reg = smf.logit(formula = 'Pool1 ~ Nrreviews_c', data = sub2).fit()
print(log_reg.summary())
###  Đánh giá mô hình logistic
X = sub2['Nrreviews_c']
y = sub2['Pool1']
# Chia dữ liệu thành 2 tập: train và test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50, shuffle=True)
log_reg2 = sm.Logit(y_train, X_train).fit()
yhat = log_reg2.predict(X_test)
prediction = list(map(round, yhat))
from sklearn.metrics import (confusion_matrix, accuracy_score)
# confusion_matrix
cm = confusion_matrix(y_test, prediction)
print ("Confusion Matrix : \n", cm)
print('Test accuracy = ', accuracy_score(y_test, prediction))