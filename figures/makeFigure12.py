# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:04:26 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:58:37 2024

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
#sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


#conversion to AF: 1 AF= 1233 m3

    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()
test_CDEC=datDaily_monthly['kwb_WRM']




data=pd.read_csv("E:/CALFEWS-main/ridge_plots/kwb_WRM_1550_1580.csv", sep='\t')
color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})

data['x']=data['x']*(1233/(10**6))
            
fig, ax = plt.subplots()

      
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

#plt.axvline(x=0, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(x=np.mean(test_CDEC)*(1233/(10**6)), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=118.98*(1233/(10**6)), color="red",ymin=0, ymax=0.05,linewidth=4)
#plt.axvline(x=np.min(test_CDEC), color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
plt.savefig('E:/CALFEWS-main/GWB/WRM_1550_1580_SI.pdf',bbox_inches='tight')


df1 = data[data['frame']=='df_baseline']
np.sum(df1['x'] == 0 ) /df1['x'].size #0.323


np.sum(test_CDEC == 0 ) /test_CDEC.size #0.323

df1 = data[data['frame']=='df_baseline']
np.sum(df1['x'] == 0 ) /df1['x'].size #0.323

df1 = data[data['frame']=='df']
np.sum(df1['x'] == 0 ) /df1['x'].size #0.323





datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()
test_CDEC=datDaily_monthly['kwb_WON']




data=pd.read_csv("E:/CALFEWS-main/ridge_plots/WON_1550_1580.csv", sep='\t')

data['x']=data['x']*(1233/(10**6))


fig, ax = plt.subplots()

color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
                  
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  
plt.axvline(x=np.mean(test_CDEC)*(1233/(10**6)), color="black",ymin=0, ymax=0.05,linewidth=4)
#plt.axvline(x=627, color="red",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=np.min(test_CDEC)*(1233/(10**6)), color="red",ymin=0, ymax=0.05,linewidth=4)
#plt.xlim(0,2500)
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
plt.savefig('E:/CALFEWS-main/GWB/WON_1550_1580_min_SI.pdf',bbox_inches='tight')


df1 = data[data['frame']=='df_baseline']
np.sum(df1['x'] == 0 ) /df1['x'].size #0.323

df1 = data[data['frame']=='df']
np.sum(df1['x'] == 0 ) /df1['x'].size #0.323
