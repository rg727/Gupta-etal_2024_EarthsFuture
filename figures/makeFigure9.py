# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:51:49 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:17:15 2024

@author: rg727
"""

import numpy as np
import pandas as pd
import h5py
import json
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress
import os
from datetime import datetime
from calfews_src import *
from calfews_src.visualizer import Visualizer
from calfews_src.util import get_results_sensitivity_number_outside_model

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.cm as cm
import os
from matplotlib.cm import ScalarMappable
from pathlib import Path
import sys 



#Create a dave style figure 


fig, ax = plt.subplots()
data=pd.read_csv("E:/CALFEWS-main/ridge_plots/swpdelta_contract_1550_1580.csv", sep='\t')

df_CC = data[data['frame']=='df']
df_CC=df_CC.reset_index()
df_CC=df_CC.dropna()

n=30
df_CC = df_CC.groupby(np.arange(len(df_CC))//n).min()




#Create a 5 year rolling average deliveries 
df_CC_5=df_CC.rolling(window=5).sum()
df_CC_5=df_CC_5.dropna()
df_CC_5['Duration']=5

df_CC_10=df_CC.rolling(window=10).sum()
df_CC_10=df_CC_10.dropna()
df_CC_10['Duration']=10

df_CC_15=df_CC.rolling(window=15).sum()
df_CC_15=df_CC_15.dropna()
df_CC_15['Duration']=15

df_CC_20=df_CC.rolling(window=20).sum()
df_CC_20=df_CC_20.dropna()
df_CC_20['Duration']=20

df_CC_25=df_CC.rolling(window=25).sum()
df_CC_25=df_CC_25.dropna()
df_CC_25['Duration']=25

df_CC_30=df_CC.rolling(window=30).sum()
df_CC_30=df_CC_30.dropna()
df_CC_30['Duration']=30

df_CC=pd.concat([df_CC_5,df_CC_10,df_CC_15,df_CC_20,df_CC_25,df_CC_30])
df_CC['Frame']="CC"


df_baseline = data[data['frame']=='df_baseline']
df_baseline=df_baseline.reset_index()
df_baseline=df_baseline.dropna()


df_baseline = df_baseline.groupby(np.arange(len(df_baseline))//n).min()



#Create a 5 year rolling average deliveries 
df_baseline_5=df_baseline.rolling(window=5).sum()
df_baseline_5=df_baseline_5.dropna()
df_baseline_5['Duration']=5

df_baseline_10=df_baseline.rolling(window=10).sum()
df_baseline_10=df_baseline_10.dropna()
df_baseline_10['Duration']=10

df_baseline_15=df_baseline.rolling(window=15).sum()
df_baseline_15=df_baseline_15.dropna()
df_baseline_15['Duration']=15

df_baseline_20=df_baseline.rolling(window=20).sum()
df_baseline_20=df_baseline_20.dropna()
df_baseline_20['Duration']=20

df_baseline_25=df_baseline.rolling(window=25).sum()
df_baseline_25=df_baseline_25.dropna()
df_baseline_25['Duration']=25

df_baseline_30=df_baseline.rolling(window=30).sum()
df_baseline_30=df_baseline_30.dropna()
df_baseline_30['Duration']=30

df_baseline=pd.concat([df_baseline_5,df_baseline_10,df_baseline_15,df_baseline_20,df_baseline_25,df_baseline_30])
df_baseline['Frame']="baseline"

data=pd.concat([df_baseline,df_CC])

color_dict = dict({'CC':'#BC6C25',
                  'baseline':'#283618'})





datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_truncated=datDaily[datDaily["Date"].dt.month.isin([9])]

datDaily_monthly=datDaily_truncated.resample('M').mean()

test_CDEC=datDaily_monthly['tableA_contract']
test_CDEC=test_CDEC.dropna()
test_CDEC=pd.DataFrame(test_CDEC)


test_CDEC_final=pd.DataFrame(columns=['Duration', 'Historical_Mean'], index=range(4))


#Create a 5 year rolling average deliveries 
test_CDEC_5=test_CDEC.rolling(window=5).sum()
test_CDEC_5=test_CDEC_5.dropna()
test_CDEC_final.iloc[0,:]=[5,np.min(test_CDEC_5['tableA_contract'])]


test_CDEC_10=test_CDEC.rolling(window=10).sum()
test_CDEC_10=test_CDEC_10.dropna()
test_CDEC_final.iloc[1,:]=[10,np.min(test_CDEC_10['tableA_contract'])]


test_CDEC_15=test_CDEC.rolling(window=15).sum()
test_CDEC_15=test_CDEC_15.dropna()
test_CDEC_final.iloc[2,:]=[15,np.min(test_CDEC_15['tableA_contract'])]


test_CDEC_20=test_CDEC.rolling(window=20).sum()
test_CDEC_20=test_CDEC_20.dropna()
test_CDEC_final.iloc[3,:]=[20,np.min(test_CDEC_20['tableA_contract'])]


data['x']=data['x']*(1233/(10**6))
test_CDEC_final['Historical_Mean']=test_CDEC_final['Historical_Mean']*(1233/(10**6))

ax=sns.boxplot(y='x', x='Duration', data=data, hue='Frame',palette=color_dict,boxprops=dict(alpha=0.6),showfliers=False)

sns.stripplot(y='Historical_Mean', x='Duration', data=test_CDEC_final, palette=["#e9d8a6"],size=10,jitter=False)
plt.legend([],[], frameon=False)
#plt.ylim(0,55000)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_ylabel('')    
ax.set_xlabel('')

plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/swpdelta_boxplots_SI.pdf',bbox_inches='tight')


###############################################################################################################################





