#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 01:22:17 2018

@author: adeesha
"""





import pandas as pd
import scipy.stats as stat
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import Imputer
import math
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv('dengue_features_train.csv')
test_data = pd.read_csv('dengue_features_test.csv')
labels_train = pd.read_csv('dengue_labels_train.csv')
submission_data = pd.read_csv('submission_format.csv')

data = pd.concat([train_data, labels_train], axis=1, join_axes=[train_data.index])

_,i = np.unique(data.columns, return_index=True)
data = data.iloc[:,i]
data = data.dropna()

data = data.replace('sj',1)
data = data.replace('iq',2)
test_data = test_data.replace('sj',1)
test_data = test_data.replace('iq',2)


data['week_value'] = data.year *100 + data.weekofyear
test_data['week_value'] = test_data.year *100 + test_data.weekofyear

features = ['week_value','city','year','weekofyear','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm']

r_list =[]
f_list=[]
for f in features: 
    r = stat.pearsonr(data[['total_cases']],data[[f]])
    if(r[1]<0.01):
        r_list.append(r[0])
        f_list.append(f)

for i in range(len(r_list)):
    print(str(f_list[i])+"  "+str(r_list[i]))
     
print (".........................................................")

selected_f = ['week_value','city','year','reanalysis_air_temp_k','reanalysis_min_air_temp_k','reanalysis_tdtr_k','station_diur_temp_rng_c','station_min_temp_c']


linreg = LinearRegression()
linreg.fit(data[selected_f],data['total_cases'])
train_pred = linreg.predict(data[selected_f])

ceil_train_pred =[]
round_train_pred = []
for d in train_pred:
    if(d<0):
        d=0
        
    else:
        d=math.ceil(d)
        dr = round(d)
    ceil_train_pred.append(d)
    round_train_pred.append(dr)

sqrt = np.sqrt(metrics.mean_squared_error(data[['total_cases']],ceil_train_pred))
print(mean_absolute_error(data[['total_cases']],ceil_train_pred))
print(mean_absolute_error(data[['total_cases']],round_train_pred))
print(sqrt)

imputer = Imputer()
test_data = test_data[selected_f]
test_imputed = imputer.fit_transform(test_data)

test_pred = linreg.predict(test_imputed)

round_test_pred =[]
for d in test_pred:
    if(d<0):
        d=0
        
    else:
        d=round(d)
    round_test_pred.append(int(d))


    
submission_data = submission_data.drop(['total_cases'], axis=1)
round_test_pred = pd.DataFrame(round_test_pred)   

submission_data = pd.concat([submission_data, round_test_pred], axis=1, join_axes=[submission_data.index])

submission_data = submission_data.rename(columns={0:'total_cases'})

submission_data.to_csv("submission_5-19.csv",encoding='utf-8', index=False)
    
