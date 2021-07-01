# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:44:37 2020

@author: Dell
"""

import numpy
import pandas
import scipy.stats as st
import statsmodels.api as sm
import seaborn
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt
pandas.set_option('display.float_format', lambda x:'%.2f'%x)
#Đọc dữ liệu
data = pandas.read_csv('LasVegasTripAdvisorReviews-Dataset1.csv', low_memory=False)
sub1=data[(data['Hotel stars']>=3) & (data['Hotel stars']<=5)]
recode1 = {"NO":0 ,"YES":1}
sub1['Free internet1']= sub1['Free internet'].map(recode1)
sub1['Free internet1'] = pandas.to_numeric(sub1['Free internet1'], errors='coerce')
data['Helpful votes'] =pandas.to_numeric(data['Helpful votes'], errors='coerce')
data['Hotel stars'] = pandas.to_numeric(data['Hotel stars'], errors='coerce')
data['Score'] =pandas.to_numeric(data['Score'], errors='coerce')
data['Nr. reviews'] =pandas.to_numeric(data['Nr. reviews'], errors='coerce')

scat1 = seaborn.regplot(x="Nr. reviews", y="Hotel stars", data=data)
plt.xlabel('Tổng số lượng review')
plt.ylabel('Số phiếu bầu hữu ích')
plt.title('Scatterplot cho mối liên hệ giữa tổng số lượng review và số phiếu bầu hữu ích')
plt.show()
###
seaborn.factorplot(x="Free internet", y="Helpful votes", data=sub1, kind="bar", ci=None)
plt.xlabel('Casino')
plt.ylabel('Helpful votes')
plt.title('Scatterplot cho mối liên hệ giữa các khách sạn có hay không dịch  ')
plt.show()
print ('mối liên hệ giữa số phiếu bầu hữu ích và tổng số lượng của review')
print (st.pearsonr(data['Helpful votes'], data['Nr. reviews']))
################### Mô hình hồi quy tuyến tính ##########################
print ("mô hình hồi quy OLS cho mối liên hệ giữa điểm của review và khách sạn có hay không dịch vụ casino")
reg1 = smf.ols('Score ~ Casino ', data=data).fit()
print (reg1.summary())
sub1 = sub1[['Score', 'Casino']].dropna()
# group means & sd
print ("Trung bình")
ds1 = sub1.groupby('Casino').mean()
print (ds1)
print ("Độ lệch chuẩn")
ds2 = sub1.groupby('Casino').std()
print (ds2)