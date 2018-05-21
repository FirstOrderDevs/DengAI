#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 01:42:17 2018

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
from sklearn.preprocessing import PolynomialFeatures

train_data = pd.read_csv('dengue_features_train.csv')
test_data = pd.read_csv('dengue_features_test.csv')
labels_train = pd.read_csv('dengue_labels_train.csv')
submission_data = pd.read_csv('submission_format.csv')

data = pd.concat([train_data, labels_train], axis=1, join_axes=[train_data.index])

_,i = np.unique(data.columns, return_index=True)
data = data.iloc[:,i]
data = data.dropna()

data['week_value'] = data.year *100 + data.weekofyear
test_data['week_value'] = test_data.year *100 + test_data.weekofyear


#data = data.replace('sj',1)
#data = data.replace('iq',2)
#test_data = test_data.replace('sj',1)
#test_data = test_data.replace('iq',2)

data_sj = data[data['city']=='sj']
data_iq = data[data['city']=='iq']
test_data_sj = test_data[test_data['city']=='sj']
test_data_iq = test_data[test_data['city']=='iq']

total_cases_sj = data_sj[['total_cases']].values
total_cases_iq = data_iq[['total_cases']].values
features = ['week_value','year','weekofyear','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm']

r_list =[]
f_list=[]
for f in features: 
    r = stat.pearsonr(data_sj[['total_cases']],data_sj[[f]])
    if(r[1]<0.01):
        r_list.append(r[0])
        f_list.append(f)

for i in range(len(r_list)):
    print(str(f_list[i])+"  "+str(r_list[i]))
     
print (".........................................................")

selected_f = ['year','reanalysis_air_temp_k','reanalysis_min_air_temp_k','reanalysis_tdtr_k','station_diur_temp_rng_c','station_min_temp_c']
selected_f_sj = ['week_value','weekofyear','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_specific_humidity_g_per_kg']

poly_sj = PolynomialFeatures(degree=1);
poly_data_sj = poly_sj.fit_transform(data_sj[selected_f_sj]);

x= data_sj['total_cases'];

linreg_sj = LinearRegression()
linreg_sj.fit(poly_data_sj,data_sj['total_cases'])
train_pred_sj = linreg_sj.predict(poly_data_sj)

ceil_train_pred_sj =[]
round_train_pred_sj = []
for d in train_pred_sj:
    if(d<0):
        d=0

    else:
        d=math.ceil(d)
        dr = round(d)
    ceil_train_pred_sj.append(d)
    round_train_pred_sj.append(dr)

sqrt_sj = np.sqrt(metrics.mean_squared_error(data_sj[['total_cases']],ceil_train_pred_sj))
print(mean_absolute_error(data_sj[['total_cases']],ceil_train_pred_sj))
print(mean_absolute_error(data_sj[['total_cases']],round_train_pred_sj))
print(sqrt_sj)

poly_iq = PolynomialFeatures(degree=1);
poly_data_iq = poly_iq.fit_transform(data_iq[selected_f]);

linreg_iq = LinearRegression()
linreg_iq.fit(poly_data_iq,data_iq['total_cases'])
train_pred_iq = linreg_iq.predict(poly_data_iq)

ceil_train_pred_iq =[]
round_train_pred_iq = []
for d in train_pred_iq:
    if(d<0):
        d=0

    else:
        d=math.ceil(d)
        dr = round(d)
    ceil_train_pred_iq.append(d)
    round_train_pred_iq.append(dr)

sqrt_iq = np.sqrt(metrics.mean_squared_error(data_iq[['total_cases']],ceil_train_pred_iq))
print(mean_absolute_error(data_iq[['total_cases']],ceil_train_pred_iq))
print(mean_absolute_error(data_iq[['total_cases']],round_train_pred_iq))
print(sqrt_iq)

round_train_pred = round_train_pred_sj + round_train_pred_iq
sqrt = np.sqrt(metrics.mean_squared_error(data[['total_cases']],round_train_pred))
print(sqrt)

imputer = Imputer()
test_data_sj = test_data_sj[selected_f_sj]
test_imputed_sj = imputer.fit_transform(test_data_sj)

test_pred_sj = linreg_sj.predict(poly_sj.fit_transform(test_imputed_sj))

round_test_pred_sj =[]
for d in test_pred_sj:
    if(d<0):
        d=0

    else:
        d=round(d)
    round_test_pred_sj.append(int(d))

test_data_iq = test_data_iq[selected_f]
test_imputed_iq = imputer.fit_transform(test_data_iq)

test_pred_iq = linreg_iq.predict(poly_iq.fit_transform(test_imputed_iq))

round_test_pred_iq =[]
for d in test_pred_iq:
    if(d<0):
        d=0

    else:
        d=round(d)
    round_test_pred_iq.append(int(d))



round_test_pred =round_test_pred_sj+round_test_pred_iq
submission_data = submission_data.drop(['total_cases'], axis=1)
round_test_pred = pd.DataFrame(round_test_pred)

submission_data = pd.concat([submission_data, round_test_pred], axis=1, join_axes=[submission_data.index])

submission_data = submission_data.rename(columns={0:'total_cases'})

submission_data.to_csv("submission_5-22.csv",encoding='utf-8', index=False)
    
