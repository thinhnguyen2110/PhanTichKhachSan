# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
## Đọc dữ liệu
data = pandas.read_csv('LasVegasTripAdvisorReviews-Dataset1.csv', low_memory=False)
#Xem xét kích thước dữ liệu 
print(len(data))
print(len(data.columns))
#thực hiện phân tích giải thích
#Helpful Vote
print('Đếm Helpful Vote')
c1= data['Helpful votes'].value_counts().sort_index()
print(c1)
print('Tính Phần Trăm Helpful vote')
c2=data['Helpful votes'].value_counts(normalize=True).sort_index()
print(c2)
#Hotel Stars
print('Đếm Số Sao')
p1= data['Hotel stars'].value_counts().sort_index()
print(p1)
print('Tính Phần Trăm Hotel stars')
p2=data['Hotel stars'].value_counts(normalize=True).sort_index()
print(p2)
#Score
print('Đếm Score')
a1= data['Score'].value_counts().sort_index()
print(a1)
print('Tính Phần Trăm Score')
a2=data['Score'].value_counts(normalize=True).sort_index()
print(a2)
#tinh chỉnh câu hỏi 1
#print('Tinh chỉnh')
Tcc1= data[(data['Hotel stars']>=3 ) &(data['Helpful votes']>200) ]
Tcc2= Tcc1.copy()
print(Tcc2)
print('đếm cho thuộc tính Hepfulvotes ban đầu')
c1 = Tcc2['Hotel stars'].value_counts(dropna=False).sort_index()
print(c1)
# Thiết lập để pandas hiển thị tất cả các cột
pandas.set_option('display.max_columns', None)
# Thiết lập để pandas hiển thị tất cả các dòng
pandas.set_option('display.max_rows', None)
#loại bỏ dữ liệu thiếu

#print('đếm cho Helpful votes với 0 được thay bằng NAN')
#c2 = Tcc2['Helpful votes'].value_counts(dropna=False).sort_index()
#print(c2)

#đồ thị
seaborn.distplot(Tcc2["Helpful votes"].dropna(), kde=False)
plt.xlabel('Số Phiếu')
plt.ylabel('Số Sao')
plt.title('Mức độ tăng sao')
plt.show()
# Chuyển giá trị thuộc tính sang giá trị số
#data['Traveler type'] = pandas.to_numeric(data['Traveler type'], errors='coerce')

#Tinh chỉnh câu hỏi 2
#print('Tinh chỉnh câu ')
#Tcc3= data[ (data['Hotel stars']>=3) & (data['Traveler type']=='Business') ]
#Tcc4 =Tcc3.copy()
#print(Tcc4)
#Mã hóa
#recode= {1:'Business'}
#Tcc4["Recode"] = Tcc4["Traveler type"].map(recode)

#đồ thị
#Tcc4["Recode"] = Tcc4["Recode"].astype('category')
#seaborn.countplot(x="Traveler type", data=Tcc4)
#plt.xlabel('Sự lựa chọn của khách hàng cặp đôi')
#plt.ylabel('Số lượng phiếu chọn')
#plt.title('Đồ thị thể hiện sự lựa chọn khách sạn có bể bơi của các cặp đôi')
#plt.show()

model1 = smf.ols(formula='Helpful votes', data=Tcc1)
results1 = model1.fit()
print(results1.summary())

