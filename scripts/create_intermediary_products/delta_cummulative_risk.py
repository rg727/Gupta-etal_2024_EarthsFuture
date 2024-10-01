# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:53:42 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:30:50 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:00:01 2024

@author: rg727
"""
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

r=6

i=3

datDaily = get_results_sensitivity_number_outside_model("/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/baseline_ensemble/"+str(i)+"/"+str(r)+"/results.hdf5", '')
#Move date column inside
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([9,10])]
test=datDaily['delta_outflow']


#Change to dataframe
test=test.to_frame()

test['Date']=test.index
#Aggregate to monthly 
test_monthly=test.resample('M').mean()

test_monthly=test_monthly.dropna()


threshold=13.68


#cummulative risk calculation 

p = np.arange(3,93,3) 



#storage=np.zeros(30)
#length=np.arange(1550, 1580, 1)
#for i in range (len(p)):
 #   x=test_monthly[0:(p[i])]
  #  storage[i]=np.sum(x['delta_outflow'] < threshold) /x['delta_outflow'].size 
    

storage=np.zeros(30)
length=np.arange(1550, 1576, 1)
for i in range (len(p)):
    x=test_monthly[0:(p[i])]
    storage[i]=np.sum(x['delta_outflow'] < threshold)/x['delta_outflow'].size 
storage=pd.DataFrame(storage).rolling(5).mean() 
storage=storage.dropna() 


df = pd.DataFrame([length,storage.iloc[:,0]]).T
 

for i in range(2,50):
    my_file=Path("/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/baseline_ensemble/"+str(i)+"/"+str(r)+"/results.hdf5")
    if my_file.exists():
      my_file="/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/baseline_ensemble/"+str(i)+"/"+str(r)+"/results.hdf5"
      datDaily = get_results_sensitivity_number_outside_model(my_file, '')
      
      datDaily['Date']=datDaily.index
      datDaily=datDaily[datDaily["Date"].dt.month.isin([9,10])]
      test=datDaily['delta_outflow']


      #Change to dataframe
      test=test.to_frame()

      test['Date']=test.index
      #Aggregate to monthly 
      test_monthly=test.resample('M').mean()

      test_monthly=test_monthly.dropna()


      threshold=13.68


      #cummulative risk calculation 

      p = np.arange(3,93,3) 



      #storage=np.zeros(30)
      #length=np.arange(1550, 1580, 1)
      #for i in range (len(p)):
       #   x=test_monthly[0:(p[i])]
        #  storage[i]=np.sum(x['delta_outflow'] < threshold) /x['delta_outflow'].size 
      #storage=np.zeros(16)
      #length=np.arange(1550, 1580, 1)
      #for i in range (16):
        #  x=test_monthly[i:i+15]
       #   storage[i]=np.sum(x['delta_outflow'] < threshold) /x['delta_outflow'].size 
      storage=np.zeros(30)
      length=np.arange(1550, 1576, 1)
      for i in range (len(p)):
          x=test_monthly[0:(p[i])]
          storage[i]=np.sum(x['delta_outflow'] < threshold)/x['delta_outflow'].size 
      storage=pd.DataFrame(storage).rolling(5).mean() 
      storage=storage.dropna() 
       
 
          

      df1 = pd.DataFrame([length,storage.iloc[:,0]]).T
      df=pd.concat([df, df1],ignore_index=True)
    
    else:
      continue 


r=6

i=2

datDaily = get_results_sensitivity_number_outside_model("/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/4T_1CC/"+str(i)+"/"+str(r)+"/results.hdf5", '')
#Move date column inside
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([9,10])]
test=datDaily['delta_outflow']


#Change to dataframe
test=test.to_frame()

test['Date']=test.index
#Aggregate to monthly 
test_monthly=test.resample('M').mean()

test_monthly=test_monthly.dropna()


threshold=13.68


#cummulative risk calculation 

p = np.arange(3,93,3) 



storage=np.zeros(30)
length=np.arange(1550, 1576, 1)
for i in range (len(p)):
    x=test_monthly[0:(p[i])]
    storage[i]=np.sum(x['delta_outflow'] < threshold)/x['delta_outflow'].size 
storage=pd.DataFrame(storage).rolling(5).mean() 
storage=storage.dropna()    

df_CC = pd.DataFrame([length,storage.iloc[:,0]]).T
 

for i in range(2,50):
    my_file=Path("/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/4T_1CC/"+str(i)+"/"+str(r)+"/results.hdf5")
    if my_file.exists():
      my_file="/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/results/4T_1CC/"+str(i)+"/"+str(r)+"/results.hdf5"
      datDaily = get_results_sensitivity_number_outside_model(my_file, '')
      
      datDaily['Date']=datDaily.index
      datDaily=datDaily[datDaily["Date"].dt.month.isin([9,10])]
      test=datDaily['delta_outflow']


      #Change to dataframe
      test=test.to_frame()

      test['Date']=test.index
      #Aggregate to monthly 
      test_monthly=test.resample('M').mean()

      test_monthly=test_monthly.dropna()


      threshold=13.68


      #cummulative risk calculation 

      p = np.arange(3,93,3) 


      storage=np.zeros(30)
      length=np.arange(1550, 1576, 1)
      for i in range (len(p)):
          x=test_monthly[0:(p[i])]
          storage[i]=np.sum(x['delta_outflow'] < threshold)/x['delta_outflow'].size 
      storage=pd.DataFrame(storage).rolling(5).mean() 
      storage=storage.dropna() 
      
      #storage=np.zeros(30)
      #length=np.arange(1550, 1580, 1)
      #for i in range (len(p)):
       #   x=test_monthly[0:(p[i])]
        #  storage[i]=np.sum(x['delta_outflow'] < threshold) /x['delta_outflow'].size 
      #storage=np.zeros(16)
      #length=np.arange(1550, 1580, 1)
      #for i in range (16):
       #   x=test_monthly[i:i+15]
        #  storage[i]=np.sum(x['delta_outflow'] < threshold) /x['delta_outflow'].size 
       
          

      df1_CC = pd.DataFrame([length,storage.iloc[:,0]]).T
      df_CC=pd.concat([df_CC, df1_CC],ignore_index=True)
    
    else:
      continue 

data = pd.concat([df.assign(frame='df_baseline'),
                  df_CC.assign(frame='df_CC')])     
                  
                  
data.to_csv('/home/fs02/pmr82_0001/rg727/CALFEWS-main-new/ridge/4T_1CC/delta_outflow_1550_1580_fall_lineplot.csv', sep='\t')    

