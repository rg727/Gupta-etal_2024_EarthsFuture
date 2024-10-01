# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:20:50 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:02:07 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:49:45 2024

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

section_number = 21
output_folder = "/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/FDC/"

matplotlib.rcParams.update({'font.size': 20})


sns.set_style("white")

fig, ax = plt.subplots(4, 3,figsize=(15, 12),constrained_layout =True)

#fig.supxlabel('Non-Exceedance Probability')

#fig.supylabel("Monthly Average Storage (tAF)")



reservoirs=['oroville_inf','shasta_inf','millerton_inf']
areas=[3607*5280*5280,6665*5280*5280,1675*5280*5280]


i=1

datDaily = get_results_sensitivity_number_outside_model("/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/results/"+str(i)+"/results.hdf5", '')


for r in range(len(reservoirs)):


    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()

    
    test_paleo=datDaily_monthly[reservoirs[r]]
    
    test_paleo=test_paleo.to_frame()
    
    sort = np.sort(test_paleo[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[0,r].plot(exceedence, sort*1233,color="gray",label="Paleo Historical")
    ax[0,r].set_title('hello')
    
    


reservoirs=['isabella_inf','pineflat_inf','kaweah_inf']
areas=[2074*5280*5280, 1545*5280*5280,561*5280*5280]


for r in range(len(reservoirs)):


    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()

    
    test_paleo=datDaily_monthly[reservoirs[r]]
    
    test_paleo=test_paleo.to_frame()
    
    sort = np.sort(test_paleo[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[1,r].plot(exceedence, sort*1233,color="gray",label="Paleo Historical")
    ax[1,r].set_title('hello')
    
    


reservoirs=['folsom_inf','success_inf','yuba_inf']
areas=[1885*5280*5280,393*5280*5280,1108*5280*5280]


for r in range(len(reservoirs)):

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()

    
    test_paleo=datDaily_monthly[reservoirs[r]]
    
    test_paleo=test_paleo.to_frame()
    
    sort = np.sort(test_paleo[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[2,r].plot(exceedence, sort*1233,color="gray",label="Paleo Historical")
    ax[2,r].set_title('hello')
    
    




reservoirs=['exchequer_inf','donpedro_inf','newmelones_inf']
areas=[1061*5280*5280,1538*5280*5280,900*5280*5280]


for r in range(len(reservoirs)):

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()

    
    test_paleo=datDaily_monthly[reservoirs[r]]
    
    test_paleo=test_paleo.to_frame()
    
    sort = np.sort(test_paleo[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[3,r].plot(exceedence, sort*1233,color="gray",label="Paleo Historical")
    ax[3,r].set_title('hello')
    



for i in range(2,50):
    
    my_file=Path("/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/results/"+str(i)+"/results.hdf5")
    if my_file.exists():
        my_file="/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/results/"+str(i)+"/results.hdf5"
        datDaily = get_results_sensitivity_number_outside_model(my_file, '')

    reservoirs=['oroville_inf','shasta_inf','millerton_inf']
    areas=[3607*5280*5280,6665*5280*5280,1675*5280*5280]
    for r in range(len(reservoirs)):
    
    
        datDaily['Date'] =datDaily.index
    
        datDaily_monthly=datDaily.resample('M').mean()
    
        
        test_paleo=datDaily_monthly[reservoirs[r]]
        
        test_paleo=test_paleo.to_frame()
        
        sort = np.sort(test_paleo[reservoirs[r]]*areas[r]/304.8/43560)
        exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
        ax[0,r].plot(exceedence, sort*1233,color="gray",label="Paleo Historical")
        ax[0,r].set_title('hello')
        
        
    
    
    reservoirs=['isabella_inf','pineflat_inf','kaweah_inf']
    areas=[2074*5280*5280, 1545*5280*5280,561*5280*5280]
    
    
    
    for r in range(len(reservoirs)):
    
    
        datDaily['Date'] =datDaily.index
    
        datDaily_monthly=datDaily.resample('M').mean()
    
        
        test_paleo=datDaily_monthly[reservoirs[r]]
        
        test_paleo=test_paleo.to_frame()
        
        sort = np.sort(test_paleo[reservoirs[r]]*areas[r]/304.8/43560)
        exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
        ax[1,r].plot(exceedence, sort*1233,color="gray",label="Paleo Historical")
        ax[1,r].set_title('hello')
        
        
    
    
    reservoirs=['folsom_inf','success_inf','yuba_inf']
    areas=[1885*5280*5280,393*5280*5280,1108*5280*5280]
    
    
    
    for r in range(len(reservoirs)):
    
        datDaily['Date'] =datDaily.index
    
        datDaily_monthly=datDaily.resample('M').mean()
    
        
        test_paleo=datDaily_monthly[reservoirs[r]]
        
        test_paleo=test_paleo.to_frame()
        
        sort = np.sort(test_paleo[reservoirs[r]]*areas[r]/304.8/43560)
        exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
        ax[2,r].plot(exceedence, sort*1233,color="gray",label="Paleo Historical")
        ax[2,r].set_title('hello')
        
        
    
    
    
    
    reservoirs=['exchequer_inf','donpedro_inf','newmelones_inf']
    areas=[1061*5280*5280,1538*5280*5280,900*5280*5280]
    
    
    for r in range(len(reservoirs)):
    
        datDaily['Date'] =datDaily.index
    
        datDaily_monthly=datDaily.resample('M').mean()
    
        
        test_paleo=datDaily_monthly[reservoirs[r]]
        
        test_paleo=test_paleo.to_frame()
        
        sort = np.sort(test_paleo[reservoirs[r]]*areas[r]/304.8/43560)
        exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
        ax[3,r].plot(exceedence, sort*1233,color="gray",label="Paleo Historical")
        ax[3,r].set_title('hello')
        



reservoirs=['oroville_inf','shasta_inf','millerton_inf']
areas=[3607*5280*5280,6665*5280*5280,1675*5280*5280]


for r in range(len(reservoirs)):

    datDaily = get_results_sensitivity_number_outside_model('/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/FDC/CDEC_results.hdf5', '')

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()


    datDaily_monthly_truncated=datDaily_monthly['1996-10-31': '2013-09-30']


    test_CDEC=datDaily_monthly_truncated[reservoirs[r]]
    
    test_CDEC=test_CDEC.to_frame()
    
    
    datDaily = get_results_sensitivity_number_outside_model('/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/FDC/SACSMA_results.hdf5', '')

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()
    
    test_SACSMA=datDaily_monthly[reservoirs[r]]
    
    test_SACSMA=test_SACSMA.to_frame()
    
    sort= np.sort(test_SACSMA[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[0,r].plot(exceedence, sort*1233,color="black",label="SACSMA Historical")
    sort = np.sort(test_CDEC[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[0,r].plot(exceedence, sort*1233,color="red",label="CDEC Historical")
    ax[0,r].set_title('hello')


reservoirs=['isabella_inf','pineflat_inf','kaweah_inf']
areas=[2074*5280*5280, 1545*5280*5280,561*5280*5280]



for r in range(len(reservoirs)):

    datDaily = get_results_sensitivity_number_outside_model('/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/FDC/CDEC_results.hdf5', '')

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()


    datDaily_monthly_truncated=datDaily_monthly['1996-10-31': '2013-09-30']


    test_CDEC=datDaily_monthly_truncated[reservoirs[r]]
    
    test_CDEC=test_CDEC.to_frame()
    
    
    datDaily = get_results_sensitivity_number_outside_model('/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/FDC/SACSMA_results.hdf5', '')

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()
    
    test_SACSMA=datDaily_monthly[reservoirs[r]]
    
    test_SACSMA=test_SACSMA.to_frame()
    
    sort= np.sort(test_SACSMA[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[1,r].plot(exceedence, sort*1233,color="black",label="SACSMA Historical")
    sort = np.sort(test_CDEC[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[1,r].plot(exceedence, sort*1233,color="red",label="CDEC Historical")
    ax[1,r].set_title('hello')



reservoirs=['folsom_inf','success_inf','yuba_inf']
areas=[1885*5280*5280,393*5280*5280,1108*5280*5280]


for r in range(len(reservoirs)):

    datDaily = get_results_sensitivity_number_outside_model('/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/FDC/CDEC_results.hdf5', '')

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()


    datDaily_monthly_truncated=datDaily_monthly['1996-10-31': '2013-09-30']


    test_CDEC=datDaily_monthly_truncated[reservoirs[r]]
    
    test_CDEC=test_CDEC.to_frame()
    
    
    datDaily = get_results_sensitivity_number_outside_model('/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/FDC/SACSMA_results.hdf5', '')

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()
    
    test_SACSMA=datDaily_monthly[reservoirs[r]]
    
    test_SACSMA=test_SACSMA.to_frame()
    
    sort= np.sort(test_SACSMA[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[2,r].plot(exceedence, sort*1233,color="black",label="SACSMA Historical")
    sort = np.sort(test_CDEC[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[2,r].plot(exceedence, sort*1233,color="red",label="CDEC Historical")
    ax[2,r].set_title('hello')




reservoirs=['exchequer_inf','donpedro_inf','newmelones_inf']
areas=[1061*5280*5280,1538*5280*5280,900*5280*5280]


for r in range(len(reservoirs)):

    datDaily = get_results_sensitivity_number_outside_model('/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/FDC/CDEC_results.hdf5', '')

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()


    datDaily_monthly_truncated=datDaily_monthly['1996-10-31': '2013-09-30']


    test_CDEC=datDaily_monthly_truncated[reservoirs[r]]
    
    test_CDEC=test_CDEC.to_frame()
    
    
    datDaily = get_results_sensitivity_number_outside_model('/home/fs02/pmr82_0001/rg727/baseline_calfews/CALFEWS_SACSMA_paleo/FDC/SACSMA_results.hdf5', '')

    datDaily['Date'] =datDaily.index

    datDaily_monthly=datDaily.resample('M').mean()
    
    test_SACSMA=datDaily_monthly[reservoirs[r]]
    
    test_SACSMA=test_SACSMA.to_frame()
    
    sort= np.sort(test_SACSMA[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[3,r].plot(exceedence, sort*1233,color="black",label="SACSMA Historical")
    sort = np.sort(test_CDEC[reservoirs[r]]*areas[r]/304.8/43560)
    exceedence = np.arange(1.,len(sort)+1) / (len(sort) +1)
    ax[3,r].plot(exceedence, sort*1233,color="red",label="CDEC Historical")
    ax[3,r].set_title('hello')







            

plt.savefig(output_folder+'inf_new_SI.pdf')



