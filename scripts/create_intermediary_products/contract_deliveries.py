# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:23:08 2024

@author: rg727
"""
import numpy as np
import pandas as pd
import h5py
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
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
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


##########################################################################################################################################

output_folder = "/home/fs02/pmr82_0001/rg727/Figure8/4T_1CC/"

if not os.path.exists(output_folder):
  os.makedirs(output_folder)
  
r=6

i=3

datDaily = get_results_sensitivity_number_outside_model("/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/baseline_ensemble/"+str(i)+"/"+str(r)+"/results.hdf5", '')
#Move date column inside
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([9])]
test=datDaily['swp_contract']


#Change to dataframe
test=test.to_frame()

test['Date']=test.index
#Aggregate to monthly 
#test_monthly=test.resample('M').sum()
test_monthly=test
test=test_monthly['swp_contract'].values


df_baseline = pd.DataFrame(dict(x=test,index=test_monthly.index,ensemble=i))

 

for i in range(2,50):
    my_file=Path("/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/baseline_ensemble/"+str(i)+"/"+str(r)+"/results.hdf5")
    if my_file.exists():
      my_file="/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/baseline_ensemble/"+str(i)+"/"+str(r)+"/results.hdf5"
      datDaily = get_results_sensitivity_number_outside_model(my_file, '')
      
      datDaily['Date']=datDaily.index
      datDaily=datDaily[datDaily["Date"].dt.month.isin([9])]
      test=datDaily['swp_contract']

      #Change to dataframe
      test=test.to_frame()

        #Move date column inside
      test['Date']=test.index

      #Aggregate to monthly 
      test_monthly=test
      #test_monthly=test.resample('M').sum()

      test=test_monthly['swp_contract'].values
      
      df1 = pd.DataFrame(dict(x=test,index=test_monthly.index,ensemble=i))

      df_baseline=pd.concat([df_baseline, df1],ignore_index=True)
    
    else:
      continue 

r=6

i=2

datDaily = get_results_sensitivity_number_outside_model("/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/4T_1CC/"+str(i)+"/"+str(r)+"/results.hdf5", '')

datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([9])]
test=datDaily['swp_contract']

#Change to dataframe
test=test.to_frame()

#Move date column inside
test['Date']=test.index

#Aggregate to monthly 
test_monthly=test
#test_monthly=test.resample('M').sum()

test=test_monthly['swp_contract'].values


df = pd.DataFrame(dict(x=test,index=test_monthly.index,ensemble=i))

 

for i in range(2,50):
    my_file=Path("/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/4T_1CC/"+str(i)+"/"+str(r)+"/results.hdf5")
    if my_file.exists():
      my_file="/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/4T_1CC/"+str(i)+"/"+str(r)+"/results.hdf5"
      datDaily = get_results_sensitivity_number_outside_model(my_file, '')
      
      datDaily['Date']=datDaily.index
      datDaily=datDaily[datDaily["Date"].dt.month.isin([9])]
      test=datDaily['swp_contract']
      #Change to dataframe
      test=test.to_frame()

        #Move date column inside
      test['Date']=test.index

      #Aggregate to monthly 
      #test_monthly=test.resample('M').sum()
      test_monthly=test

      test=test_monthly['swp_contract'].values

      df1 = pd.DataFrame(dict(x=test,index=test_monthly.index,ensemble=i))

      df=pd.concat([df, df1],ignore_index=True)
    
    else:
      continue 

###########################################################################################################################################
   

data = pd.concat([df.assign(frame='df'),
                  df_baseline.assign(frame='df_baseline')])     
                  
                  
data.to_csv('swpdelta_contract_1550_1580.csv')    



