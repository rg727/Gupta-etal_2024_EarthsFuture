# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:48:23 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:42:59 2024

@author: rg727
"""
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

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



datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

df = pd.DataFrame(columns=['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'], index=range(1))

months=np.array([10,11,12,1,2,3,4,5,6,7,8,9])

for i in range(len(months)):
    datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

    datDaily['Date']=datDaily.index

    datDaily=datDaily[datDaily["Date"].dt.month.isin([months[i]])]
    test=datDaily['shasta_S']
    df.iloc[0,i]=np.nanmean(test)


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

df_drought = pd.DataFrame(columns=['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'], index=range(1))

months=np.array([10,11,12,1,2,3,4,5,6,7,8,9])

for i in range(len(months)):
    datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

    datDaily['Date']=datDaily.index
    datDaily=datDaily[datDaily["Date"].dt.month.isin([months[i]])]
    datDaily=datDaily['2011-10-31': '2016-09-30']
    test=datDaily['shasta_S']
    df_drought.iloc[0,i]=np.nanmin(test)





ensemble=pd.read_csv("E:/CALFEWS-main/quantiles/shasta_s_quantiles.csv",sep='\t')
ensemble=ensemble.iloc[:,1:13]

ensemble=ensemble*(1233/(10**6))


fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(ensemble, ps[j], axis=0)
    l = np.percentile(ensemble, ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')


ax.plot(np.arange(12), df.iloc[0,:]*(1233/(10**6)), linestyle='--', color='black', linewidth=2)
ax.plot(np.arange(12), df_drought.iloc[0,:]*(1233/(10**6)), linestyle='--', color='red', linewidth=2)
ax.axhline(550*(1233/(10**6)), linestyle='-', color='gray', linewidth=2)
ax.axhline(4552*(1233/(10**6)), linestyle='-', color='gray', linewidth=2)



ax.set_xlim(0,11)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('E:/CALFEWS-main/quantiles/shasta_mins_SI.pdf')




datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

df = pd.DataFrame(columns=['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'], index=range(1))

months=np.array([10,11,12,1,2,3,4,5,6,7,8,9])

for i in range(len(months)):
    datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

    datDaily['Date']=datDaily.index

    datDaily=datDaily[datDaily["Date"].dt.month.isin([months[i]])]
    test=datDaily['oroville_S']
    df.iloc[0,i]=np.nanmean(test)


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

df_drought = pd.DataFrame(columns=['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'], index=range(1))

months=np.array([10,11,12,1,2,3,4,5,6,7,8,9])

for i in range(len(months)):
    datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

    datDaily['Date']=datDaily.index
    datDaily=datDaily[datDaily["Date"].dt.month.isin([months[i]])]
    datDaily=datDaily['2011-10-31': '2016-09-30']
    test=datDaily['oroville_S']
    df_drought.iloc[0,i]=np.nanmin(test)
    
    
    
    
    

ensemble=pd.read_csv("E:/CALFEWS-main/quantiles/oroville_s_quantiles.csv",sep='\t')
ensemble=ensemble.iloc[:,1:13]
ensemble=ensemble*(1233/(10**6))

fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(ensemble, ps[j], axis=0)
    l = np.percentile(ensemble, ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')


ax.plot(np.arange(12), df.iloc[0,:]*(1233/(10**6)), linestyle='--', color='black', linewidth=2)
ax.plot(np.arange(12), df_drought.iloc[0,:]*(1233/(10**6)), linestyle='--', color='red', linewidth=2)
ax.axhline(852*(1233/(10**6)), linestyle='-', color='gray', linewidth=2)
ax.axhline(3537*(1233/(10**6)), linestyle='-', color='gray', linewidth=2)



ax.set_xlim(0,11)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('E:/CALFEWS-main/quantiles/oroville_mins_SI.pdf')







