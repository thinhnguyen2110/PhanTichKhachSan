# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:55:35 2020

@author: Dell
"""

import numpy
import pandas
import statsmodels.api as sm
import seaborn
import statsmodels.formula.api as smf 
import matplotlib.pyplot as plt

data = pandas.read_csv('LasVegasTripAdvisorReviews-Dataset1.csv', low_memory=False)
# Col = ['UserCountry','NrReviews','NrHotelReviews','HelpfulVotes','Score','PeriodOfStay','TravelerType',
#         'Pool','Gym','TennisCourt','Spa','Casino','FreeInternet','HotelName','HotelStars','NrRooms',
#         'UserContinent','MemberYears','ReviewMonth','ReviewWeekday']
sub1=data[(data['Hotel stars']>=3) & (data['Hotel stars']<=5)]
recode1 = {"NO":0 ,"YES":1}
sub1['Casino1']= sub1['Casino'].map(recode1)
sub1['Casino1'] = pandas.to_numeric(sub1['Casino1'], errors='coerce')

print('Mô hình hồi quy tuyến tính ')
reg2 = smf.ols('Score ~ Casino1 + Casino1 + Pool', data=sub1).fit()
print (reg2.summary())

print('Mo hinh hồi quy đa thức với 2 bien')
reg3 = smf.ols('Score  ~ Casino1 + I(Casino1**2) + Pool', data=sub1).fit()
print (reg3.summary())

#Đồ thị

sub1 = data[['Casino','Helpful votes', 'Nr. hotel reviews']].dropna()
scat1 = seaborn.regplot(x="Nr. hotel reviews", y="Helpful votes", scatter=True, data=sub1)
scat1 = seaborn.regplot(x="Nr. hotel reviews", y="Helpful votes", scatter=True, order=2, data=sub1)
plt.xlabel('Tổng số Review Khách Sạn ')
plt.ylabel('Số phiếu bầu hữu ích')
plt.show()