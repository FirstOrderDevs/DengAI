#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:37:34 2018

@author: adeesha
"""

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.linear_model import LassoCV
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pylab as plt2


train_data = pd.read_csv('/home/adeesha/repos/DengAI/dengue_features_train.csv')
test_data = pd.read_csv('/home/adeesha/repos/DengAI/dengue_features_test.csv')
labels_train = pd.read_csv('/home/adeesha/repos/DengAI/dengue_labels_train.csv')
submission_data = pd.read_csv('/home/adeesha/repos/DengAI/submission_format.csv')

data = pd.concat([train_data, labels_train], axis=1, join_axes=[train_data.index])

_,i = np.unique(data.columns, return_index=True)
data = data.iloc[:,i]
data = data.dropna(axis=0)

data.week_start_date = pd.to_datetime(data.week_start_date)
test_data.week_start_date = pd.to_datetime(test_data.week_start_date)

data_sj = data[data['city']=='sj']
data_iq = data[data['city']=='iq']
test_data_sj = test_data[test_data['city']=='sj']
test_data_iq = test_data[test_data['city']=='iq']

features = ['year','weekofyear','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm']

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
low_varience = sel.fit_transform(data_sj[features])


