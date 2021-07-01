# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 21:14:23 2020

@author: Dell
"""

import numpy
import pandas
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
data = pandas.read_csv('LasVegasTripAdvisorReviews-Dataset1.csv', low_memory=False)
#Có phải Spa liên quan đến số phiếu bầu hữu ích không?
data['Hotel stars'] = pandas.to_numeric(data['Hotel stars'], errors='coerce')
data['Helpful votes'] = pandas.to_numeric(data['Helpful votes'], errors='coerce')
sub1=data[(data['Hotel stars']>=3) & (data['Hotel stars']<=5)]
recode1 = {"NO":0 ,"YES":1}

sub1['Gym1']= sub1['Gym'].map(recode1)
sub1['Gym1'] = pandas.to_numeric(sub1['Gym1'], errors='coerce')
ct1 = sub1.groupby('Gym1').size()
print (ct1)
print("========Phân tích ANOVA============")
model1 = smf.ols(formula='Gym1 ~ C(Score)', data=sub1)
results1 = model1.fit()
print(results1.summary())
print("Giá trị trung bình của Spa1 liên quan đến điểm của các phiếu bầu")
sub2 = sub1[['Gym1', 'Score']].dropna()
m1= sub2.groupby('Score').mean()
print(m1)
print("Độ lệch chuẩn của Gym1 liên quan đến điểm của các phiếu bầu")
sd1 = sub2.groupby('Score').std()
print(sd1)
# Phân tích ANOVA cho thấy, F = 0.3679 với p = 0.832 (Với mức ý nghĩa là 0.05). Điều đó ta không thể bác bỏ giả
#thuyết vô hiệu rằng những khách sạn có DV Spa không có liên quan đến việc chấm điểm cho các bình luận khảo sát.

#Biến giải thích nhiều hơn 2 loại
# ========= Biến giải thích nhiều hơn 2 loại
#recode1 = {1: "Friends", 2: "Business", 3: "Families", 4: "Friends"}
#sub1['Traveler type']= sub1['Hotel stars'].map(recode1)
sub3 = sub1[['Score', 'Traveler type']].dropna()
print(sub3['Traveler type'].value_counts().sort_index())
## Phân tích sâu
mc1 = multi.MultiComparison(sub3['Score'], sub3['Traveler type'])
res1 = mc1.tukeyhsd()
print(res1.summary())

mh2 = smf.ols(formula='Score ~ C(Traveler type)', data=sub3).fit()
print(mh2.summary())

print (' trung bình cho numcigmo_est theo chủng tộc')
m2= sub3.groupby('Traveler type').mean()
print (m2)

print (' độ lệch chuẩn cho for numcigmo_est theo chủng tộc')
sd2 = sub3.groupby('Traveler type').std()
print (sd2)

