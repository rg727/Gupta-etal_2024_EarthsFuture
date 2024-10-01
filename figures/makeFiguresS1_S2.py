# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:36:26 2024

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

#############################################################################
def convert_to_WY(row):
    if row['Date'].month>=10:
        return(pd.datetime(row['Date'].year+1,1,1).year)
    else:
        return(pd.datetime(row['Date'].year,1,1).year)
    

##########################################################################################################################################

def convert_to_WY_paleo(row):
    if row['month']>=10:
        return(row['year']+1)
    else:
        return(row['year'])
    
#########################################################################################################################
#Year Array

years=np.arange(1400,2018)

years_array=np.tile(years,50)


CDEC_years=np.arange(1997,2014)


#########################################################################################################################

#read in CALFEWS CDEC baseline

calfews_dataset=pd.read_csv("E:/CALFEWS-main/hydrology/calfews_src-data.csv")


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')


calfews_dataset.index=datDaily.index


calfews_dataset['Date']= datDaily.index

calfews_dataset['WaterYear'] = calfews_dataset.apply(lambda x: convert_to_WY(x), axis=1)



#Aggregate to accumulated WY flow

calfews_dataset_annual=calfews_dataset.groupby(['WaterYear']).mean()
calfews_dataset_annual=calfews_dataset_annual.iloc[0:17,:]

CDEC_north=pd.DataFrame(calfews_dataset_annual['SHA_fnf']+calfews_dataset_annual['ORO_fnf']+calfews_dataset_annual['FOL_fnf']+calfews_dataset_annual['YRS_fnf']+calfews_dataset_annual['NML_fnf']+calfews_dataset_annual['DNP_fnf']+calfews_dataset_annual['EXC_fnf'])
CDEC_south=pd.DataFrame(calfews_dataset_annual['MIL_fnf']+calfews_dataset_annual['KWH_fnf']+calfews_dataset_annual['SUC_fnf']+calfews_dataset_annual['PFT_fnf']+calfews_dataset_annual['ISB_fnf'])

CDEC_north["Year"]=CDEC_years
CDEC_north.columns=['Streamflow','Year']


##############################################################################

#Calculate 3 year, 5-year, 10-year mins for multiple subsections of the 1500s 



CDEC_north_daily=pd.DataFrame(calfews_dataset['SHA_fnf']+calfews_dataset['ORO_fnf']+calfews_dataset['FOL_fnf']+calfews_dataset['YRS_fnf']+calfews_dataset['NML_fnf']+calfews_dataset['DNP_fnf']+calfews_dataset['EXC_fnf'])
CDEC_south_daily=pd.DataFrame(calfews_dataset['MIL_fnf']+calfews_dataset['KWH_fnf']+calfews_dataset['SUC_fnf']+calfews_dataset['PFT_fnf']+calfews_dataset['ISB_fnf'])


CDEC_north_daily_three=np.nanmin(CDEC_north_daily.rolling(3*365).mean())
CDEC_north_daily_five=np.nanmin(CDEC_north_daily.rolling(5*365).mean())
CDEC_north_daily_ten=np.nanmin(CDEC_north_daily.rolling(10*365).mean())

CDEC_south_daily_three=np.nanmin(CDEC_south_daily.rolling(3*365).mean())
CDEC_south_daily_five=np.nanmin(CDEC_south_daily.rolling(5*365).mean())
CDEC_south_daily_ten=np.nanmin(CDEC_south_daily.rolling(10*365).mean())



#SHA paleo 


SHA_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SHA_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_SHA = 6665*5280*5280


i=6
df=pd.DataFrame(SHA_paleo.iloc[44163:55121,i]*area_ft2_SHA/304.8/43560)
df
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[44163:55121,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1520_1550 = df


i=6
df=pd.DataFrame(SHA_paleo.iloc[55120:66078,i]*area_ft2_SHA/304.8/43560)
df
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[55120:66078,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1550_1580 = df



i=6
df=pd.DataFrame(SHA_paleo.iloc[47815:58772,i]*area_ft2_SHA/304.8/43560)
df
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[47815:58772,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1530_1560 = df


i=6
df=pd.DataFrame(SHA_paleo.iloc[51468:62425,i]*area_ft2_SHA/304.8/43560)
df
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[51468:62425,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1540_1570 = df



#ORO paleo

ORO_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ORO_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_ORO = 3607*5280*5280

i=6
df=pd.DataFrame(ORO_paleo.iloc[44163:55121,i]*area_ft2_ORO/304.8/43560)
df
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[44163:55121,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1520_1550 = df


i=6
df=pd.DataFrame(ORO_paleo.iloc[55120:66078,i]*area_ft2_ORO/304.8/43560)
df
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[55120:66078,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1550_1580 = df



i=6
df=pd.DataFrame(ORO_paleo.iloc[47815:58772,i]*area_ft2_ORO/304.8/43560)
df
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[47815:58772,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1530_1560 = df


i=6
df=pd.DataFrame(ORO_paleo.iloc[51468:62425,i]*area_ft2_ORO/304.8/43560)
df
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[51468:62425,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1540_1570 = df

#FOL paleo

FOL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/FOL_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_FOL = 1885*5280*5280



i=6
df=pd.DataFrame(FOL_paleo.iloc[44163:55121,i]*area_ft2_FOL/304.8/43560)
df
df.columns=['FOL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(FOL_paleo.iloc[44163:55121,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_FOL_1520_1550 = df


i=6
df=pd.DataFrame(FOL_paleo.iloc[55120:66078,i]*area_ft2_FOL/304.8/43560)
df
df.columns=['FOL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(FOL_paleo.iloc[55120:66078,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_FOL_1550_1580 = df



i=6
df=pd.DataFrame(FOL_paleo.iloc[47815:58772,i]*area_ft2_FOL/304.8/43560)
df
df.columns=['FOL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(FOL_paleo.iloc[47815:58772,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_FOL_1530_1560 = df


i=6
df=pd.DataFrame(FOL_paleo.iloc[51468:62425,i]*area_ft2_FOL/304.8/43560)
df
df.columns=['FOL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(FOL_paleo.iloc[51468:62425,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_FOL_1540_1570 = df

#DNP paleo

DNP_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TLG_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_DNP = 1538*5280*5280


i=6
df=pd.DataFrame(DNP_paleo.iloc[44163:55121,i]*area_ft2_DNP/304.8/43560)
df
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[44163:55121,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1520_1550 = df


i=6
df=pd.DataFrame(DNP_paleo.iloc[55120:66078,i]*area_ft2_DNP/304.8/43560)
df
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[55120:66078,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1550_1580 = df



i=6
df=pd.DataFrame(DNP_paleo.iloc[47815:58772,i]*area_ft2_DNP/304.8/43560)
df
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[47815:58772,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1530_1560 = df


i=6
df=pd.DataFrame(DNP_paleo.iloc[51468:62425,i]*area_ft2_DNP/304.8/43560)
df
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[51468:62425,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1540_1570 = df

#NML paleo

NML_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/NML_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_NML = 900*5280*5280


i=6
df=pd.DataFrame(NML_paleo.iloc[44163:55121,i]*area_ft2_NML/304.8/43560)
df
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[44163:55121,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1520_1550 = df


i=6
df=pd.DataFrame(NML_paleo.iloc[55120:66078,i]*area_ft2_NML/304.8/43560)
df
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[55120:66078,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1550_1580 = df



i=6
df=pd.DataFrame(NML_paleo.iloc[47815:58772,i]*area_ft2_NML/304.8/43560)
df
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[47815:58772,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1530_1560 = df


i=6
df=pd.DataFrame(NML_paleo.iloc[51468:62425,i]*area_ft2_NML/304.8/43560)
df
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[51468:62425,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1540_1570 = df

YRS_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/YRS_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_YRS = 1108*5280*5280



i=6
df=pd.DataFrame(YRS_paleo.iloc[44163:55121,i]*area_ft2_YRS/304.8/43560)
df
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[44163:55121,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1520_1550 = df


i=6
df=pd.DataFrame(YRS_paleo.iloc[55120:66078,i]*area_ft2_YRS/304.8/43560)
df
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[55120:66078,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1550_1580 = df



i=6
df=pd.DataFrame(YRS_paleo.iloc[47815:58772,i]*area_ft2_YRS/304.8/43560)
df
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[47815:58772,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1530_1560 = df


i=6
df=pd.DataFrame(YRS_paleo.iloc[51468:62425,i]*area_ft2_YRS/304.8/43560)
df
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[51468:62425,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1540_1570 = df


#EXC paleo

EXC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/EXC_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_EXC = 1061*5280*5280



i=6
df=pd.DataFrame(EXC_paleo.iloc[44163:55121,i]*area_ft2_EXC/304.8/43560)
df
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[44163:55121,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1520_1550 = df


i=6
df=pd.DataFrame(EXC_paleo.iloc[55120:66078,i]*area_ft2_EXC/304.8/43560)
df
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[55120:66078,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1550_1580 = df



i=6
df=pd.DataFrame(EXC_paleo.iloc[47815:58772,i]*area_ft2_EXC/304.8/43560)
df
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[47815:58772,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1530_1560 = df


i=6
df=pd.DataFrame(EXC_paleo.iloc[51468:62425,i]*area_ft2_EXC/304.8/43560)
df
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[51468:62425,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1540_1570 = df




#MIL paleo

MIL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/MIL_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_MIL = 1675*5280*5280


i=6
df=pd.DataFrame(MIL_paleo.iloc[44163:55121,i]*area_ft2_MIL/304.8/43560)
df
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[44163:55121,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1520_1550 = df


i=6
df=pd.DataFrame(MIL_paleo.iloc[55120:66078,i]*area_ft2_MIL/304.8/43560)
df
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[55120:66078,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1550_1580 = df



i=6
df=pd.DataFrame(MIL_paleo.iloc[47815:58772,i]*area_ft2_MIL/304.8/43560)
df
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[47815:58772,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1530_1560 = df


i=6
df=pd.DataFrame(MIL_paleo.iloc[51468:62425,i]*area_ft2_MIL/304.8/43560)
df
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[51468:62425,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1540_1570 = df


#ISB paleo

ISB_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ISB_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_ISB = 2074*5280*5280

i=6
df=pd.DataFrame(ISB_paleo.iloc[44163:55121,i]*area_ft2_ISB/304.8/43560)
df
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[44163:55121,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1520_1550 = df


i=6
df=pd.DataFrame(ISB_paleo.iloc[55120:66078,i]*area_ft2_ISB/304.8/43560)
df
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[55120:66078,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1550_1580 = df



i=6
df=pd.DataFrame(ISB_paleo.iloc[47815:58772,i]*area_ft2_ISB/304.8/43560)
df
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[47815:58772,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1530_1560 = df


i=6
df=pd.DataFrame(ISB_paleo.iloc[51468:62425,i]*area_ft2_ISB/304.8/43560)
df
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[51468:62425,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1540_1570 = df


#SUC paleo

SUC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SUC_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_SUC = 393*5280*5280


i=6
df=pd.DataFrame(SUC_paleo.iloc[44163:55121,i]*area_ft2_SUC/304.8/43560)
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[44163:55121,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_SUC = df

i=6
df=pd.DataFrame(SUC_paleo.iloc[44163:55121,i]*area_ft2_SUC/304.8/43560)
df
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[44163:55121,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SUC_1520_1550 = df


i=6
df=pd.DataFrame(SUC_paleo.iloc[55120:66078,i]*area_ft2_SUC/304.8/43560)
df
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[55120:66078,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SUC_1550_1580 = df



i=6
df=pd.DataFrame(SUC_paleo.iloc[47815:58772,i]*area_ft2_SUC/304.8/43560)
df
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[47815:58772,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SUC_1530_1560 = df


i=6
df=pd.DataFrame(SUC_paleo.iloc[51468:62425,i]*area_ft2_SUC/304.8/43560)
df
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[51468:62425,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SUC_1540_1570 = df

#KWH paleo

KWH_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TRM_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_KWH = 561*5280*5280


i=6
df=pd.DataFrame(KWH_paleo.iloc[44163:55121,i]*area_ft2_KWH/304.8/43560)
df
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[44163:55121,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1520_1550 = df


i=6
df=pd.DataFrame(KWH_paleo.iloc[55120:66078,i]*area_ft2_KWH/304.8/43560)
df
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[55120:66078,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1550_1580 = df



i=6
df=pd.DataFrame(KWH_paleo.iloc[47815:58772,i]*area_ft2_KWH/304.8/43560)
df
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[47815:58772,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1530_1560 = df


i=6
df=pd.DataFrame(KWH_paleo.iloc[51468:62425,i]*area_ft2_KWH/304.8/43560)
df
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[51468:62425,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1540_1570 = df


#PNF paleo

PNF_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/PNF_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_PNF = 1545*5280*5280


i=6
df=pd.DataFrame(PNF_paleo.iloc[44163:55121,i]*area_ft2_PNF/304.8/43560)
df
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[44163:55121,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1520_1550 = df


i=6
df=pd.DataFrame(PNF_paleo.iloc[55120:66078,i]*area_ft2_PNF/304.8/43560)
df
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[55120:66078,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1550_1580 = df



i=6
df=pd.DataFrame(PNF_paleo.iloc[47815:58772,i]*area_ft2_PNF/304.8/43560)
df
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[47815:58772,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1530_1560 = df


i=6
df=pd.DataFrame(PNF_paleo.iloc[51468:62425,i]*area_ft2_PNF/304.8/43560)
df
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[51468:62425,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1540_1570 = df



paleo_north_1520_1550=data_SHA_1520_1550.iloc[:,0]+data_ORO_1520_1550.iloc[:,0]+data_YRS_1520_1550.iloc[:,0]+data_FOL_1520_1550.iloc[:,0]+data_EXC_1520_1550.iloc[:,0]+data_NML_1520_1550.iloc[:,0]+data_DNP_1520_1550.iloc[:,0]
paleo_south_1520_1550=data_MIL_1520_1550.iloc[:,0]+data_SUC_1520_1550.iloc[:,0]+data_ISB_1520_1550.iloc[:,0]+data_KWH_1520_1550.iloc[:,0]+data_PNF_1520_1550.iloc[:,0]



paleo_north_1550_1580=data_SHA_1550_1580.iloc[:,0]+data_ORO_1550_1580.iloc[:,0]+data_YRS_1550_1580.iloc[:,0]+data_FOL_1550_1580.iloc[:,0]+data_EXC_1550_1580.iloc[:,0]+data_NML_1550_1580.iloc[:,0]+data_DNP_1550_1580.iloc[:,0]
paleo_south_1550_1580=data_MIL_1550_1580.iloc[:,0]+data_SUC_1550_1580.iloc[:,0]+data_ISB_1550_1580.iloc[:,0]+data_KWH_1550_1580.iloc[:,0]+data_PNF_1550_1580.iloc[:,0]


paleo_north_1530_1560=data_SHA_1530_1560.iloc[:,0]+data_ORO_1530_1560.iloc[:,0]+data_YRS_1530_1560.iloc[:,0]+data_FOL_1530_1560.iloc[:,0]+data_EXC_1530_1560.iloc[:,0]+data_NML_1530_1560.iloc[:,0]+data_DNP_1530_1560.iloc[:,0]
paleo_south_1530_1560=data_MIL_1530_1560.iloc[:,0]+data_SUC_1530_1560.iloc[:,0]+data_ISB_1530_1560.iloc[:,0]+data_KWH_1530_1560.iloc[:,0]+data_PNF_1530_1560.iloc[:,0]


paleo_north_1540_1570=data_SHA_1540_1570.iloc[:,0]+data_ORO_1540_1570.iloc[:,0]+data_YRS_1540_1570.iloc[:,0]+data_FOL_1540_1570.iloc[:,0]+data_EXC_1540_1570.iloc[:,0]+data_NML_1540_1570.iloc[:,0]+data_DNP_1540_1570.iloc[:,0]
paleo_south_1540_1570=data_MIL_1540_1570.iloc[:,0]+data_SUC_1540_1570.iloc[:,0]+data_ISB_1540_1570.iloc[:,0]+data_KWH_1540_1570.iloc[:,0]+data_PNF_1540_1570.iloc[:,0]



##########################################calculate rolling min####################################

#3 year rolling mean 


n=30*365

paleo_north_three_rolling=pd.DataFrame(paleo_north_1520_1550.rolling(3*365).mean())
paleo_north_three_1520_1550=paleo_north_three_rolling.groupby(np.arange(len(paleo_north_three_rolling))//n).min()

paleo_north_three_rolling=pd.DataFrame(paleo_north_1530_1560.rolling(3*365).mean())
paleo_north_three_1530_1560=paleo_north_three_rolling.groupby(np.arange(len(paleo_north_three_rolling))//n).min()



paleo_north_three_rolling=pd.DataFrame(paleo_north_1540_1570.rolling(3*365).mean())
paleo_north_three_1540_1570=paleo_north_three_rolling.groupby(np.arange(len(paleo_north_three_rolling))//n).min()

paleo_north_three_rolling=pd.DataFrame(paleo_north_1550_1580.rolling(3*365).mean())
paleo_north_three_1550_1580=paleo_north_three_rolling.groupby(np.arange(len(paleo_north_three_rolling))//n).min()



paleo_south_three_rolling=pd.DataFrame(paleo_south_1520_1550.rolling(3*365).mean())
paleo_south_three_1520_1550=paleo_south_three_rolling.groupby(np.arange(len(paleo_south_three_rolling))//n).min()

paleo_south_three_rolling=pd.DataFrame(paleo_south_1530_1560.rolling(3*365).mean())
paleo_south_three_1530_1560=paleo_south_three_rolling.groupby(np.arange(len(paleo_south_three_rolling))//n).min()



paleo_south_three_rolling=pd.DataFrame(paleo_south_1540_1570.rolling(3*365).mean())
paleo_south_three_1540_1570=paleo_south_three_rolling.groupby(np.arange(len(paleo_south_three_rolling))//n).min()

paleo_south_three_rolling=pd.DataFrame(paleo_south_1550_1580.rolling(3*365).mean())
paleo_south_three_1550_1580=paleo_south_three_rolling.groupby(np.arange(len(paleo_south_three_rolling))//n).min()



## combine these different collections into a list
data_to_plot = [np.array(paleo_north_three_1520_1550.iloc[:,0]*1233),np.array(paleo_north_three_1530_1560.iloc[:,0]*1233),np.array(paleo_north_three_1540_1570.iloc[:,0]*1233),np.array(paleo_north_three_1550_1580.iloc[:,0]*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        

ax.plot(1, CDEC_north_daily_three*1233, marker='o' ,color='#e9d8a6') 
ax.plot(2, CDEC_north_daily_three*1233, marker='o', color='#e9d8a6') 
ax.plot(3, CDEC_north_daily_three*1233, marker='o',color='#e9d8a6' ) 
ax.plot(4, CDEC_north_daily_three*1233, marker='o',color='#e9d8a6' ) 
labels = ['1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Three Year-North')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/threeyear_north_SI.pdf',bbox_inches='tight')





## combine these different collections into a list
data_to_plot = [np.array(paleo_south_three_1520_1550.iloc[:,0]*1233),np.array(paleo_south_three_1530_1560.iloc[:,0]*1233),np.array(paleo_south_three_1540_1570.iloc[:,0]*1233),np.array(paleo_south_three_1550_1580.iloc[:,0]*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        

ax.plot(1, CDEC_south_daily_three*1233, marker='o' ,color='#e9d8a6') 
ax.plot(2, CDEC_south_daily_three*1233, marker='o', color='#e9d8a6') 
ax.plot(3, CDEC_south_daily_three*1233, marker='o',color='#e9d8a6' ) 
ax.plot(4, CDEC_south_daily_three*1233, marker='o',color='#e9d8a6' ) 
labels = ['1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Three Year-South')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/threeyear_south_SI.pdf',bbox_inches='tight')



#5 year rolling mean 

n=30*365

paleo_north_five_rolling=pd.DataFrame(paleo_north_1520_1550.rolling(5*365).mean())
paleo_north_five_1520_1550=paleo_north_five_rolling.groupby(np.arange(len(paleo_north_five_rolling))//n).min()

paleo_north_five_rolling=pd.DataFrame(paleo_north_1530_1560.rolling(5*365).mean())
paleo_north_five_1530_1560=paleo_north_five_rolling.groupby(np.arange(len(paleo_north_five_rolling))//n).min()



paleo_north_five_rolling=pd.DataFrame(paleo_north_1540_1570.rolling(5*365).mean())
paleo_north_five_1540_1570=paleo_north_five_rolling.groupby(np.arange(len(paleo_north_five_rolling))//n).min()

paleo_north_five_rolling=pd.DataFrame(paleo_north_1550_1580.rolling(5*365).mean())
paleo_north_five_1550_1580=paleo_north_five_rolling.groupby(np.arange(len(paleo_north_five_rolling))//n).min()



paleo_south_five_rolling=pd.DataFrame(paleo_south_1520_1550.rolling(5*365).mean())
paleo_south_five_1520_1550=paleo_south_five_rolling.groupby(np.arange(len(paleo_south_five_rolling))//n).min()

paleo_south_five_rolling=pd.DataFrame(paleo_south_1530_1560.rolling(5*365).mean())
paleo_south_five_1530_1560=paleo_south_five_rolling.groupby(np.arange(len(paleo_south_five_rolling))//n).min()



paleo_south_five_rolling=pd.DataFrame(paleo_south_1540_1570.rolling(5*365).mean())
paleo_south_five_1540_1570=paleo_south_five_rolling.groupby(np.arange(len(paleo_south_five_rolling))//n).min()

paleo_south_five_rolling=pd.DataFrame(paleo_south_1550_1580.rolling(5*365).mean())
paleo_south_five_1550_1580=paleo_south_five_rolling.groupby(np.arange(len(paleo_south_five_rolling))//n).min()



## combine these different collections into a list
data_to_plot = [np.array(paleo_north_five_1520_1550.iloc[:,0]*1233),np.array(paleo_north_five_1530_1560.iloc[:,0]*1233),np.array(paleo_north_five_1540_1570.iloc[:,0]*1233),np.array(paleo_north_five_1550_1580.iloc[:,0]*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        

ax.plot(1, CDEC_north_daily_five*1233, marker='o' ,color='#e9d8a6') 
ax.plot(2, CDEC_north_daily_five*1233, marker='o', color='#e9d8a6') 
ax.plot(3, CDEC_north_daily_five*1233, marker='o',color='#e9d8a6' ) 
ax.plot(4, CDEC_north_daily_five*1233, marker='o',color='#e9d8a6' ) 
labels = ['1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Five Year-North')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/fiveyear_north_SI.pdf',bbox_inches='tight')





## combine these different collections into a list
data_to_plot = [np.array(paleo_south_five_1520_1550.iloc[:,0]*1233),np.array(paleo_south_five_1530_1560.iloc[:,0]*1233),np.array(paleo_south_five_1540_1570.iloc[:,0]*1233),np.array(paleo_south_five_1550_1580.iloc[:,0]*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        

ax.plot(1, CDEC_south_daily_five*1233, marker='o' ,color='#e9d8a6') 
ax.plot(2, CDEC_south_daily_five*1233, marker='o', color='#e9d8a6') 
ax.plot(3, CDEC_south_daily_five*1233, marker='o',color='#e9d8a6' ) 
ax.plot(4, CDEC_south_daily_five*1233, marker='o',color='#e9d8a6' ) 
labels = ['1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Five Year-South')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/fiveyear_south_SI.pdf',bbox_inches='tight')


#10-Year Rolling Mean 


n=30*365

paleo_north_ten_rolling=pd.DataFrame(paleo_north_1520_1550.rolling(10*365).mean())
paleo_north_ten_1520_1550=paleo_north_ten_rolling.groupby(np.arange(len(paleo_north_ten_rolling))//n).min()

paleo_north_ten_rolling=pd.DataFrame(paleo_north_1530_1560.rolling(10*365).mean())
paleo_north_ten_1530_1560=paleo_north_ten_rolling.groupby(np.arange(len(paleo_north_ten_rolling))//n).min()



paleo_north_ten_rolling=pd.DataFrame(paleo_north_1540_1570.rolling(10*365).mean())
paleo_north_ten_1540_1570=paleo_north_ten_rolling.groupby(np.arange(len(paleo_north_ten_rolling))//n).min()

paleo_north_ten_rolling=pd.DataFrame(paleo_north_1550_1580.rolling(10*365).mean())
paleo_north_ten_1550_1580=paleo_north_ten_rolling.groupby(np.arange(len(paleo_north_ten_rolling))//n).min()



paleo_south_ten_rolling=pd.DataFrame(paleo_south_1520_1550.rolling(10*365).mean())
paleo_south_ten_1520_1550=paleo_south_ten_rolling.groupby(np.arange(len(paleo_south_ten_rolling))//n).min()

paleo_south_ten_rolling=pd.DataFrame(paleo_south_1530_1560.rolling(10*365).mean())
paleo_south_ten_1530_1560=paleo_south_ten_rolling.groupby(np.arange(len(paleo_south_ten_rolling))//n).min()



paleo_south_ten_rolling=pd.DataFrame(paleo_south_1540_1570.rolling(10*365).mean())
paleo_south_ten_1540_1570=paleo_south_ten_rolling.groupby(np.arange(len(paleo_south_ten_rolling))//n).min()

paleo_south_ten_rolling=pd.DataFrame(paleo_south_1550_1580.rolling(10*365).mean())
paleo_south_ten_1550_1580=paleo_south_ten_rolling.groupby(np.arange(len(paleo_south_ten_rolling))//n).min()



## combine these different collections into a list
data_to_plot = [np.array(paleo_north_ten_1520_1550.iloc[:,0]*1233),np.array(paleo_north_ten_1530_1560.iloc[:,0]*1233),np.array(paleo_north_ten_1540_1570.iloc[:,0]*1233),np.array(paleo_north_ten_1550_1580.iloc[:,0]*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        

ax.plot(1, CDEC_north_daily_ten*1233, marker='o' ,color='#e9d8a6') 
ax.plot(2, CDEC_north_daily_ten*1233, marker='o', color='#e9d8a6') 
ax.plot(3, CDEC_north_daily_ten*1233, marker='o',color='#e9d8a6' ) 
ax.plot(4, CDEC_north_daily_ten*1233, marker='o',color='#e9d8a6' ) 
labels = ['1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Ten Year-North')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/tenyear_north_SI.pdf',bbox_inches='tight')





## combine these different collections into a list
data_to_plot = [np.array(paleo_south_ten_1520_1550.iloc[:,0]*1233),np.array(paleo_south_ten_1530_1560.iloc[:,0]*1233),np.array(paleo_south_ten_1540_1570.iloc[:,0]*1233),np.array(paleo_south_ten_1550_1580.iloc[:,0]*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        

ax.plot(1, CDEC_south_daily_ten*1233, marker='o' ,color='#e9d8a6') 
ax.plot(2, CDEC_south_daily_ten*1233, marker='o', color='#e9d8a6') 
ax.plot(3, CDEC_south_daily_ten*1233, marker='o',color='#e9d8a6' ) 
ax.plot(4, CDEC_south_daily_ten*1233, marker='o',color='#e9d8a6' ) 
labels = ['1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Ten Year-South')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/tenyear_south_SI.pdf',bbox_inches='tight')


###################################### Monthly Minimum #######################################################


calfews_dataset=pd.read_csv("E:/CALFEWS-main/hydrology/calfews_src-data.csv")


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')


calfews_dataset.index=datDaily.index


calfews_dataset['Date']= datDaily.index

datDaily_monthly=calfews_dataset.resample('M').min()
datDaily_monthly_truncated=datDaily_monthly['1996-10-31': '2013-09-30']


CDEC_north=pd.DataFrame(datDaily_monthly_truncated['SHA_fnf']+datDaily_monthly_truncated['ORO_fnf']+datDaily_monthly_truncated['FOL_fnf']+datDaily_monthly_truncated['YRS_fnf']+datDaily_monthly_truncated['NML_fnf']+datDaily_monthly_truncated['DNP_fnf']+datDaily_monthly_truncated['EXC_fnf'])
CDEC_south=pd.DataFrame(datDaily_monthly_truncated['MIL_fnf']+datDaily_monthly_truncated['KWH_fnf']+datDaily_monthly_truncated['SUC_fnf']+datDaily_monthly_truncated['PFT_fnf']+datDaily_monthly_truncated['ISB_fnf'])
###################################################################################


#SHA paleo 


SHA_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SHA_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_SHA = 6665*5280*5280


i=6
df=pd.DataFrame(SHA_paleo.iloc[44163:55121,i]*area_ft2_SHA/304.8/43560)
date=pd.date_range(start="1730-10-01",end="1760-09-30")
df.index=date

df=df.resample('M').min()
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[44163:55121,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1520_1550 = df


i=6
df=pd.DataFrame(SHA_paleo.iloc[55120:66078,i]*area_ft2_SHA/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['SHA_fnf']




for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[55120:66078,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1550_1580 = df


date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(SHA_paleo.iloc[47815:58772,i]*area_ft2_SHA/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[47815:58772,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1530_1560 = df


i=6
df=pd.DataFrame(SHA_paleo.iloc[51468:62425,i]*area_ft2_SHA/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[51468:62425,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1540_1570 = df



#ORO paleo

ORO_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ORO_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_ORO = 3607*5280*5280
date=pd.date_range(start="1730-10-01",end="1760-09-30")


i=6
df=pd.DataFrame(ORO_paleo.iloc[44163:55121,i]*area_ft2_ORO/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[44163:55121,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1520_1550 = df


i=6
df=pd.DataFrame(ORO_paleo.iloc[55120:66078,i]*area_ft2_ORO/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[55120:66078,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1550_1580 = df

date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(ORO_paleo.iloc[47815:58772,i]*area_ft2_ORO/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[47815:58772,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1530_1560 = df


i=6
df=pd.DataFrame(ORO_paleo.iloc[51468:62425,i]*area_ft2_ORO/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[51468:62425,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1540_1570 = df

#FOL paleo

FOL_paleo = pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/FOL_baseline.txt", sep='\t')

#Aggregate to accumulated WY flow

area_ft2_FOL = 1885*5280*5280

date = pd.date_range(start="1730-10-01", end="1760-09-30")


i = 6
df = pd.DataFrame(FOL_paleo.iloc[44163:55121, i]*area_ft2_FOL/304.8/43560)
df.index = date
df = df.resample('M').min()
df.columns = ['FOL_fnf']


for i in range(7, 56):
    df1 = pd.DataFrame(FOL_paleo.iloc[44163:55121, i]*area_ft2_FOL/304.8/43560)
    df1.columns = ['FOL_fnf']
    df1.index = date
    df1 = df1.resample('M').min()
    df = pd.concat([df, df1], ignore_index=True)

data_FOL_1520_1550 = df


i = 6
df = pd.DataFrame(FOL_paleo.iloc[55120:66078, i]*area_ft2_FOL/304.8/43560)
df.index = date
df = df.resample('M').min()
df.columns = ['FOL_fnf']


for i in range(7, 56):
    df1 = pd.DataFrame(FOL_paleo.iloc[55120:66078, i]*area_ft2_FOL/304.8/43560)
    df1.columns = ['FOL_fnf']
    df1.index = date
    df1 = df1.resample('M').min()
    df = pd.concat([df, df1], ignore_index=True)

data_FOL_1550_1580 = df

date = pd.date_range(start="1760-10-01", end="1790-09-30")


i = 6
df = pd.DataFrame(FOL_paleo.iloc[47815:58772, i]*area_ft2_FOL/304.8/43560)
df.index = date
df = df.resample('M').min()
df.columns = ['FOL_fnf']


for i in range(7, 56):
    df1 = pd.DataFrame(FOL_paleo.iloc[47815:58772, i]*area_ft2_FOL/304.8/43560)
    df1.columns = ['FOL_fnf']
    df1.index = date
    df1 = df1.resample('M').min()
    df = pd.concat([df, df1], ignore_index=True)

data_FOL_1530_1560 = df


i = 6
df=pd.DataFrame(FOL_paleo.iloc[51468:62425,i]*area_ft2_FOL/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['FOL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(FOL_paleo.iloc[51468:62425,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_FOL_1540_1570 = df

#DNP paleo

DNP_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TLG_baseline.txt",sep='\t')

date=pd.date_range(start="1730-10-01",end="1760-09-30")
#Aggregate to accumulated WY flow

area_ft2_DNP = 1538*5280*5280


i=6
df=pd.DataFrame(DNP_paleo.iloc[44163:55121,i]*area_ft2_DNP/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[44163:55121,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1520_1550 = df


i=6
df=pd.DataFrame(DNP_paleo.iloc[55120:66078,i]*area_ft2_DNP/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[55120:66078,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1550_1580 = df


date=pd.date_range(start="1760-10-01",end="1790-09-30")
i=6
df=pd.DataFrame(DNP_paleo.iloc[47815:58772,i]*area_ft2_DNP/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[47815:58772,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1530_1560 = df


i=6
df=pd.DataFrame(DNP_paleo.iloc[51468:62425,i]*area_ft2_DNP/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[51468:62425,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1540_1570 = df

#NML paleo

NML_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/NML_baseline.txt",sep='\t')
date=pd.date_range(start="1730-10-01",end="1760-09-30")
#Aggregate to accumulated WY flow

area_ft2_NML = 900*5280*5280


i=6
df=pd.DataFrame(NML_paleo.iloc[44163:55121,i]*area_ft2_NML/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[44163:55121,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1520_1550 = df


i=6
df=pd.DataFrame(NML_paleo.iloc[55120:66078,i]*area_ft2_NML/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[55120:66078,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1550_1580 = df

date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(NML_paleo.iloc[47815:58772,i]*area_ft2_NML/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[47815:58772,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1530_1560 = df


i=6
df=pd.DataFrame(NML_paleo.iloc[51468:62425,i]*area_ft2_NML/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[51468:62425,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1540_1570 = df

YRS_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/YRS_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_YRS = 1108*5280*5280
date=pd.date_range(start="1730-10-01",end="1760-09-30")


i=6
df=pd.DataFrame(YRS_paleo.iloc[44163:55121,i]*area_ft2_YRS/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[44163:55121,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1520_1550 = df


i=6
df=pd.DataFrame(YRS_paleo.iloc[55120:66078,i]*area_ft2_YRS/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[55120:66078,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1550_1580 = df

date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(YRS_paleo.iloc[47815:58772,i]*area_ft2_YRS/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[47815:58772,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1530_1560 = df


i=6
df=pd.DataFrame(YRS_paleo.iloc[51468:62425,i]*area_ft2_YRS/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[51468:62425,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1540_1570 = df


#EXC paleo

EXC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/EXC_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_EXC = 1061*5280*5280
date=pd.date_range(start="1730-10-01",end="1760-09-30")


i=6
df=pd.DataFrame(EXC_paleo.iloc[44163:55121,i]*area_ft2_EXC/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[44163:55121,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1520_1550 = df


i=6
df=pd.DataFrame(EXC_paleo.iloc[55120:66078,i]*area_ft2_EXC/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[55120:66078,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1550_1580 = df

date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(EXC_paleo.iloc[47815:58772,i]*area_ft2_EXC/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[47815:58772,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1530_1560 = df


i=6
df=pd.DataFrame(EXC_paleo.iloc[51468:62425,i]*area_ft2_EXC/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[51468:62425,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1540_1570 = df




#MIL paleo

MIL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/MIL_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_MIL = 1675*5280*5280
date=pd.date_range(start="1730-10-01",end="1760-09-30")

i=6
df=pd.DataFrame(MIL_paleo.iloc[44163:55121,i]*area_ft2_MIL/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[44163:55121,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1520_1550 = df


i=6
df=pd.DataFrame(MIL_paleo.iloc[55120:66078,i]*area_ft2_MIL/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[55120:66078,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1550_1580 = df

date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(MIL_paleo.iloc[47815:58772,i]*area_ft2_MIL/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[47815:58772,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1530_1560 = df


i=6
df=pd.DataFrame(MIL_paleo.iloc[51468:62425,i]*area_ft2_MIL/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[51468:62425,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1540_1570 = df


#ISB paleo

ISB_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ISB_baseline.txt",sep='\t')
date=pd.date_range(start="1730-10-01",end="1760-09-30")
#Aggregate to accumulated WY flow

area_ft2_ISB = 2074*5280*5280

i=6
df=pd.DataFrame(ISB_paleo.iloc[44163:55121,i]*area_ft2_ISB/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[44163:55121,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1520_1550 = df


i=6
df=pd.DataFrame(ISB_paleo.iloc[55120:66078,i]*area_ft2_ISB/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[55120:66078,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1550_1580 = df

date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(ISB_paleo.iloc[47815:58772,i]*area_ft2_ISB/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[47815:58772,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1530_1560 = df


i=6
df=pd.DataFrame(ISB_paleo.iloc[51468:62425,i]*area_ft2_ISB/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[51468:62425,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1540_1570 = df


#SUC paleo

SUC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SUC_baseline.txt",sep='\t')
date=pd.date_range(start="1730-10-01",end="1760-09-30")
#Aggregate to accumulated WY flow

area_ft2_SUC = 393*5280*5280


i=6
df=pd.DataFrame(SUC_paleo.iloc[44163:55121,i]*area_ft2_SUC/304.8/43560)
df.index=date

df=df.resample('M').min()
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[44163:55121,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)
    

data_SUC = df

i=6
df=pd.DataFrame(SUC_paleo.iloc[44163:55121,i]*area_ft2_SUC/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[44163:55121,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_SUC_1520_1550 = df


i=6
df=pd.DataFrame(SUC_paleo.iloc[55120:66078,i]*area_ft2_SUC/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[55120:66078,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_SUC_1550_1580 = df

date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(SUC_paleo.iloc[47815:58772,i]*area_ft2_SUC/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[47815:58772,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_SUC_1530_1560 = df


i=6
df=pd.DataFrame(SUC_paleo.iloc[51468:62425,i]*area_ft2_SUC/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[51468:62425,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_SUC_1540_1570 = df

#KWH paleo

KWH_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TRM_baseline.txt",sep='\t')
date=pd.date_range(start="1730-10-01",end="1760-09-30")
#Aggregate to accumulated WY flow

area_ft2_KWH = 561*5280*5280


i=6
df=pd.DataFrame(KWH_paleo.iloc[44163:55121,i]*area_ft2_KWH/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[44163:55121,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1520_1550 = df


i=6
df=pd.DataFrame(KWH_paleo.iloc[55120:66078,i]*area_ft2_KWH/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[55120:66078,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1550_1580 = df

date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(KWH_paleo.iloc[47815:58772,i]*area_ft2_KWH/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[47815:58772,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1530_1560 = df


i=6
df=pd.DataFrame(KWH_paleo.iloc[51468:62425,i]*area_ft2_KWH/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[51468:62425,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1540_1570 = df


#PNF paleo

PNF_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/PNF_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow
date=pd.date_range(start="1730-10-01",end="1760-09-30")
area_ft2_PNF = 1545*5280*5280


i=6
df=pd.DataFrame(PNF_paleo.iloc[44163:55121,i]*area_ft2_PNF/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[44163:55121,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1520_1550 = df


i=6
df=pd.DataFrame(PNF_paleo.iloc[55120:66078,i]*area_ft2_PNF/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[55120:66078,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1550_1580 = df

date=pd.date_range(start="1760-10-01",end="1790-09-30")

i=6
df=pd.DataFrame(PNF_paleo.iloc[47815:58772,i]*area_ft2_PNF/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[47815:58772,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1530_1560 = df


i=6
df=pd.DataFrame(PNF_paleo.iloc[51468:62425,i]*area_ft2_PNF/304.8/43560)
df.index=date
df=df.resample('M').min()
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[51468:62425,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df1.index=date
    df1=df1.resample('M').min()
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1540_1570 = df



paleo_north_1520_1550=data_SHA_1520_1550.iloc[:,0]+data_ORO_1520_1550.iloc[:,0]+data_YRS_1520_1550.iloc[:,0]+data_FOL_1520_1550.iloc[:,0]+data_EXC_1520_1550.iloc[:,0]+data_NML_1520_1550.iloc[:,0]+data_DNP_1520_1550.iloc[:,0]
paleo_south_1520_1550=data_MIL_1520_1550.iloc[:,0]+data_SUC_1520_1550.iloc[:,0]+data_ISB_1520_1550.iloc[:,0]+data_KWH_1520_1550.iloc[:,0]+data_PNF_1520_1550.iloc[:,0]



paleo_north_1550_1580=data_SHA_1550_1580.iloc[:,0]+data_ORO_1550_1580.iloc[:,0]+data_YRS_1550_1580.iloc[:,0]+data_FOL_1550_1580.iloc[:,0]+data_EXC_1550_1580.iloc[:,0]+data_NML_1550_1580.iloc[:,0]+data_DNP_1550_1580.iloc[:,0]
paleo_south_1550_1580=data_MIL_1550_1580.iloc[:,0]+data_SUC_1550_1580.iloc[:,0]+data_ISB_1550_1580.iloc[:,0]+data_KWH_1550_1580.iloc[:,0]+data_PNF_1550_1580.iloc[:,0]


paleo_north_1530_1560=data_SHA_1530_1560.iloc[:,0]+data_ORO_1530_1560.iloc[:,0]+data_YRS_1530_1560.iloc[:,0]+data_FOL_1530_1560.iloc[:,0]+data_EXC_1530_1560.iloc[:,0]+data_NML_1530_1560.iloc[:,0]+data_DNP_1530_1560.iloc[:,0]
paleo_south_1530_1560=data_MIL_1530_1560.iloc[:,0]+data_SUC_1530_1560.iloc[:,0]+data_ISB_1530_1560.iloc[:,0]+data_KWH_1530_1560.iloc[:,0]+data_PNF_1530_1560.iloc[:,0]


paleo_north_1540_1570=data_SHA_1540_1570.iloc[:,0]+data_ORO_1540_1570.iloc[:,0]+data_YRS_1540_1570.iloc[:,0]+data_FOL_1540_1570.iloc[:,0]+data_EXC_1540_1570.iloc[:,0]+data_NML_1540_1570.iloc[:,0]+data_DNP_1540_1570.iloc[:,0]
paleo_south_1540_1570=data_MIL_1540_1570.iloc[:,0]+data_SUC_1540_1570.iloc[:,0]+data_ISB_1540_1570.iloc[:,0]+data_KWH_1540_1570.iloc[:,0]+data_PNF_1540_1570.iloc[:,0]



## combine these different collections into a list
data_to_plot = [np.array(CDEC_north.iloc[:,0]*1233),np.array(paleo_north_1520_1550*1233),np.array(paleo_north_1530_1560*1233),np.array(paleo_north_1540_1570*1233),np.array(paleo_north_1550_1580*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        


labels = ['Historical','1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4,5]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yscale('log')
plt.title('Monthly Minimum-North')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/monthly_minimum_north_SI.pdf',bbox_inches='tight')




## combine these different collections into a list
data_to_plot = [np.array(CDEC_south.iloc[:,0]*1233),np.array(paleo_south_1520_1550*1233),np.array(paleo_south_1530_1560*1233),np.array(paleo_south_1540_1570*1233),np.array(paleo_south_1550_1580*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        


labels = ['Historical','1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4,5]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yscale('log')
plt.title('Monthly Minimum-south')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/monthly_minimum_south_SI.pdf',bbox_inches='tight')



###########################Annual Accumulated Flow##########################################################

def convert_to_WY_paleo(row):
    if row['month']>=10:
        return(row['year']+1)
    else:
        return(row['year'])
    


SHA_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SHA_baseline.txt",sep='\t')
SHA_paleo = SHA_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
SHA_paleo['WaterYear'] = SHA_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_SHA = 6665*5280*5280


i=7
SHA_paleo_median_yearly_accum=SHA_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(SHA_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_SHA/304.8/43560)

df.columns=['SHA_fnf']



for i in range(8,56):
    df1=pd.DataFrame(SHA_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1520_1550 = df


i=7
SHA_paleo_median_yearly_accum=SHA_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(SHA_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_SHA/304.8/43560)

df.columns=['SHA_fnf']



for i in range(8,56):
    df1=pd.DataFrame(SHA_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SHA_1550_1580 = df




i=7

df=pd.DataFrame(SHA_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_SHA/304.8/43560)

df.columns=['SHA_fnf']



for i in range(8,56):
    df1=pd.DataFrame(SHA_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_SHA_1530_1560 = df


i=7
df=pd.DataFrame(SHA_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_SHA/304.8/43560)

df.columns=['SHA_fnf']



for i in range(8,56):
    df1=pd.DataFrame(SHA_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_SHA_1540_1570 = df




ORO_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ORO_baseline.txt",sep='\t')
ORO_paleo = ORO_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
ORO_paleo['WaterYear'] = ORO_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_ORO = 3607*5280*5280


i=7
ORO_paleo_median_yearly_accum=ORO_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(ORO_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_ORO/304.8/43560)

df.columns=['ORO_fnf']



for i in range(8,56):
    df1=pd.DataFrame(ORO_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1520_1550 = df


i=7
ORO_paleo_median_yearly_accum=ORO_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(ORO_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_ORO/304.8/43560)

df.columns=['ORO_fnf']



for i in range(8,56):
    df1=pd.DataFrame(ORO_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ORO_1550_1580 = df




i=7

df=pd.DataFrame(ORO_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_ORO/304.8/43560)

df.columns=['ORO_fnf']



for i in range(8,56):
    df1=pd.DataFrame(ORO_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_ORO_1530_1560 = df


i=7
df=pd.DataFrame(ORO_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_ORO/304.8/43560)

df.columns=['ORO_fnf']



for i in range(8,56):
    df1=pd.DataFrame(ORO_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_ORO_1540_1570 = df





FOL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/FOL_baseline.txt",sep='\t')
FOL_paleo = FOL_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
FOL_paleo['WaterYear'] = FOL_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow


area_ft2_FOL = 1885*5280*5280


i=7
FOL_paleo_median_yearly_accum=FOL_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(FOL_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_FOL/304.8/43560)

df.columns=['FOL_fnf']



for i in range(8,56):
    df1=pd.DataFrame(FOL_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_FOL_1520_1550 = df


i=7
FOL_paleo_median_yearly_accum=FOL_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(FOL_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_FOL/304.8/43560)

df.columns=['FOL_fnf']



for i in range(8,56):
    df1=pd.DataFrame(FOL_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_FOL_1550_1580 = df




i=7

df=pd.DataFrame(FOL_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_FOL/304.8/43560)

df.columns=['FOL_fnf']



for i in range(8,56):
    df1=pd.DataFrame(FOL_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_FOL_1530_1560 = df


i=7
df=pd.DataFrame(FOL_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_FOL/304.8/43560)

df.columns=['FOL_fnf']



for i in range(8,56):
    df1=pd.DataFrame(FOL_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_FOL_1540_1570 = df



DNP_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TLG_baseline.txt",sep='\t')
DNP_paleo = DNP_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
DNP_paleo['WaterYear'] = DNP_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow


area_ft2_DNP = 1538*5280*5280


i=7
DNP_paleo_median_yearly_accum=DNP_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(DNP_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_DNP/304.8/43560)

df.columns=['DNP_fnf']



for i in range(8,56):
    df1=pd.DataFrame(DNP_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1520_1550 = df


i=7
DNP_paleo_median_yearly_accum=DNP_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(DNP_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_DNP/304.8/43560)

df.columns=['DNP_fnf']



for i in range(8,56):
    df1=pd.DataFrame(DNP_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_DNP_1550_1580 = df




i=7

df=pd.DataFrame(DNP_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_DNP/304.8/43560)

df.columns=['DNP_fnf']



for i in range(8,56):
    df1=pd.DataFrame(DNP_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_DNP_1530_1560 = df


i=7
df=pd.DataFrame(DNP_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_DNP/304.8/43560)

df.columns=['DNP_fnf']



for i in range(8,56):
    df1=pd.DataFrame(DNP_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_DNP_1540_1570 = df




NML_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/NML_baseline.txt",sep='\t')
NML_paleo = NML_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
NML_paleo['WaterYear'] = NML_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow


area_ft2_NML = 900*5280*5280


i=7
NML_paleo_median_yearly_accum=NML_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(NML_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_NML/304.8/43560)

df.columns=['NML_fnf']



for i in range(8,56):
    df1=pd.DataFrame(NML_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1520_1550 = df


i=7
NML_paleo_median_yearly_accum=NML_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(NML_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_NML/304.8/43560)

df.columns=['NML_fnf']



for i in range(8,56):
    df1=pd.DataFrame(NML_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_NML_1550_1580 = df




i=7

df=pd.DataFrame(NML_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_NML/304.8/43560)

df.columns=['NML_fnf']



for i in range(8,56):
    df1=pd.DataFrame(NML_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_NML_1530_1560 = df


i=7
df=pd.DataFrame(NML_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_NML/304.8/43560)

df.columns=['NML_fnf']



for i in range(8,56):
    df1=pd.DataFrame(NML_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_NML_1540_1570 = df






YRS_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/YRS_baseline.txt",sep='\t')
YRS_paleo = YRS_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
YRS_paleo['WaterYear'] = YRS_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow


area_ft2_YRS = 1108*5280*5280


i=7
YRS_paleo_median_yearly_accum=YRS_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(YRS_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_YRS/304.8/43560)

df.columns=['YRS_fnf']



for i in range(8,56):
    df1=pd.DataFrame(YRS_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1520_1550 = df


i=7
YRS_paleo_median_yearly_accum=YRS_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(YRS_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_YRS/304.8/43560)

df.columns=['YRS_fnf']



for i in range(8,56):
    df1=pd.DataFrame(YRS_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_YRS_1550_1580 = df




i=7

df=pd.DataFrame(YRS_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_YRS/304.8/43560)

df.columns=['YRS_fnf']



for i in range(8,56):
    df1=pd.DataFrame(YRS_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_YRS_1530_1560 = df


i=7
df=pd.DataFrame(YRS_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_YRS/304.8/43560)

df.columns=['YRS_fnf']



for i in range(8,56):
    df1=pd.DataFrame(YRS_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_YRS_1540_1570 = df





EXC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/EXC_baseline.txt",sep='\t')
EXC_paleo = EXC_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
EXC_paleo['WaterYear'] = EXC_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow


area_ft2_EXC = 1061*5280*5280


i=7
EXC_paleo_median_yearly_accum=EXC_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(EXC_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_EXC/304.8/43560)

df.columns=['EXC_fnf']



for i in range(8,56):
    df1=pd.DataFrame(EXC_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1520_1550 = df


i=7
EXC_paleo_median_yearly_accum=EXC_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(EXC_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_EXC/304.8/43560)

df.columns=['EXC_fnf']



for i in range(8,56):
    df1=pd.DataFrame(EXC_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_EXC_1550_1580 = df




i=7

df=pd.DataFrame(EXC_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_EXC/304.8/43560)

df.columns=['EXC_fnf']



for i in range(8,56):
    df1=pd.DataFrame(EXC_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_EXC_1530_1560 = df


i=7
df=pd.DataFrame(EXC_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_EXC/304.8/43560)

df.columns=['EXC_fnf']



for i in range(8,56):
    df1=pd.DataFrame(EXC_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_EXC_1540_1570 = df




MIL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/MIL_baseline.txt",sep='\t')
MIL_paleo = MIL_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
MIL_paleo['WaterYear'] = MIL_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow


area_ft2_MIL = 1675*5280*5280


i=7
MIL_paleo_median_yearly_accum=MIL_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(MIL_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_MIL/304.8/43560)

df.columns=['MIL_fnf']



for i in range(8,56):
    df1=pd.DataFrame(MIL_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1520_1550 = df


i=7
MIL_paleo_median_yearly_accum=MIL_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(MIL_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_MIL/304.8/43560)

df.columns=['MIL_fnf']



for i in range(8,56):
    df1=pd.DataFrame(MIL_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_MIL_1550_1580 = df




i=7

df=pd.DataFrame(MIL_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_MIL/304.8/43560)

df.columns=['MIL_fnf']



for i in range(8,56):
    df1=pd.DataFrame(MIL_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_MIL_1530_1560 = df


i=7
df=pd.DataFrame(MIL_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_MIL/304.8/43560)

df.columns=['MIL_fnf']



for i in range(8,56):
    df1=pd.DataFrame(MIL_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_MIL_1540_1570 = df




ISB_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ISB_baseline.txt",sep='\t')
ISB_paleo = ISB_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
ISB_paleo['WaterYear'] = ISB_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow


area_ft2_ISB = 2074*5280*5280


i=7
ISB_paleo_median_yearly_accum=ISB_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(ISB_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_ISB/304.8/43560)

df.columns=['ISB_fnf']



for i in range(8,56):
    df1=pd.DataFrame(ISB_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1520_1550 = df


i=7
ISB_paleo_median_yearly_accum=ISB_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(ISB_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_ISB/304.8/43560)

df.columns=['ISB_fnf']



for i in range(8,56):
    df1=pd.DataFrame(ISB_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_ISB_1550_1580 = df




i=7

df=pd.DataFrame(ISB_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_ISB/304.8/43560)

df.columns=['ISB_fnf']



for i in range(8,56):
    df1=pd.DataFrame(ISB_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_ISB_1530_1560 = df


i=7
df=pd.DataFrame(ISB_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_ISB/304.8/43560)

df.columns=['ISB_fnf']



for i in range(8,56):
    df1=pd.DataFrame(ISB_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_ISB_1540_1570 = df






KWH_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TRM_baseline.txt",sep='\t')
KWH_paleo = KWH_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
KWH_paleo['WaterYear'] = KWH_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow


area_ft2_KWH = 561*5280*5280


i=7
KWH_paleo_median_yearly_accum=KWH_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(KWH_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_KWH/304.8/43560)

df.columns=['KWH_fnf']



for i in range(8,56):
    df1=pd.DataFrame(KWH_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1520_1550 = df


i=7
KWH_paleo_median_yearly_accum=KWH_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(KWH_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_KWH/304.8/43560)

df.columns=['KWH_fnf']



for i in range(8,56):
    df1=pd.DataFrame(KWH_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_KWH_1550_1580 = df




i=7

df=pd.DataFrame(KWH_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_KWH/304.8/43560)

df.columns=['KWH_fnf']



for i in range(8,56):
    df1=pd.DataFrame(KWH_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_KWH_1530_1560 = df


i=7
df=pd.DataFrame(KWH_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_KWH/304.8/43560)

df.columns=['KWH_fnf']



for i in range(8,56):
    df1=pd.DataFrame(KWH_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_KWH_1540_1570 = df



PNF_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/PNF_baseline.txt",sep='\t')
PNF_paleo = PNF_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
PNF_paleo['WaterYear'] = PNF_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow


area_ft2_PNF = 1545*5280*5280


i=7
PNF_paleo_median_yearly_accum=PNF_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(PNF_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_PNF/304.8/43560)

df.columns=['PNF_fnf']



for i in range(8,56):
    df1=pd.DataFrame(PNF_paleo_median_yearly_accum.iloc[120:150,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1520_1550 = df


i=7
PNF_paleo_median_yearly_accum=PNF_paleo.groupby(['WaterYear'],as_index=False).sum()

df=pd.DataFrame(PNF_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_PNF/304.8/43560)

df.columns=['PNF_fnf']



for i in range(8,56):
    df1=pd.DataFrame(PNF_paleo_median_yearly_accum.iloc[150:180,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_PNF_1550_1580 = df



i=7

df=pd.DataFrame(PNF_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_PNF/304.8/43560)

df.columns=['PNF_fnf']



for i in range(8,56):
    df1=pd.DataFrame(PNF_paleo_median_yearly_accum.iloc[130:160,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_PNF_1530_1560 = df


i=7
df=pd.DataFrame(PNF_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_PNF/304.8/43560)

df.columns=['PNF_fnf']



for i in range(8,56):
    df1=pd.DataFrame(PNF_paleo_median_yearly_accum.iloc[140:170,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)


data_PNF_1540_1570 = df


paleo_north_1520_1550=data_SHA_1520_1550.iloc[:,0]+data_ORO_1520_1550.iloc[:,0]+data_YRS_1520_1550.iloc[:,0]+data_FOL_1520_1550.iloc[:,0]+data_EXC_1520_1550.iloc[:,0]+data_NML_1520_1550.iloc[:,0]+data_DNP_1520_1550.iloc[:,0]
paleo_south_1520_1550=data_MIL_1520_1550.iloc[:,0]+data_SUC_1520_1550.iloc[:,0]+data_ISB_1520_1550.iloc[:,0]+data_KWH_1520_1550.iloc[:,0]+data_PNF_1520_1550.iloc[:,0]



paleo_north_1550_1580=data_SHA_1550_1580.iloc[:,0]+data_ORO_1550_1580.iloc[:,0]+data_YRS_1550_1580.iloc[:,0]+data_FOL_1550_1580.iloc[:,0]+data_EXC_1550_1580.iloc[:,0]+data_NML_1550_1580.iloc[:,0]+data_DNP_1550_1580.iloc[:,0]
paleo_south_1550_1580=data_MIL_1550_1580.iloc[:,0]+data_SUC_1550_1580.iloc[:,0]+data_ISB_1550_1580.iloc[:,0]+data_KWH_1550_1580.iloc[:,0]+data_PNF_1550_1580.iloc[:,0]


paleo_north_1530_1560=data_SHA_1530_1560.iloc[:,0]+data_ORO_1530_1560.iloc[:,0]+data_YRS_1530_1560.iloc[:,0]+data_FOL_1530_1560.iloc[:,0]+data_EXC_1530_1560.iloc[:,0]+data_NML_1530_1560.iloc[:,0]+data_DNP_1530_1560.iloc[:,0]
paleo_south_1530_1560=data_MIL_1530_1560.iloc[:,0]+data_SUC_1530_1560.iloc[:,0]+data_ISB_1530_1560.iloc[:,0]+data_KWH_1530_1560.iloc[:,0]+data_PNF_1530_1560.iloc[:,0]


paleo_north_1540_1570=data_SHA_1540_1570.iloc[:,0]+data_ORO_1540_1570.iloc[:,0]+data_YRS_1540_1570.iloc[:,0]+data_FOL_1540_1570.iloc[:,0]+data_EXC_1540_1570.iloc[:,0]+data_NML_1540_1570.iloc[:,0]+data_DNP_1540_1570.iloc[:,0]
paleo_south_1540_1570=data_MIL_1540_1570.iloc[:,0]+data_SUC_1540_1570.iloc[:,0]+data_ISB_1540_1570.iloc[:,0]+data_KWH_1540_1570.iloc[:,0]+data_PNF_1540_1570.iloc[:,0]





## combine these different collections into a list
data_to_plot = [np.array(CDEC_north.iloc[:,0]*1233),np.array(paleo_north_1520_1550*1233),np.array(paleo_north_1530_1560*1233),np.array(paleo_north_1540_1570*1233),np.array(paleo_north_1550_1580*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        


labels = ['Historical','1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4,5]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yscale('log')
plt.title('Annual Accumulated-North')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/annual_north_SI.pdf',bbox_inches='tight')




## combine these different collections into a list
data_to_plot = [np.array(CDEC_south.iloc[:,0]*1233),np.array(paleo_south_1520_1550*1233),np.array(paleo_south_1530_1560*1233),np.array(paleo_south_1540_1570*1233),np.array(paleo_south_1550_1580*1233)]

# Create a figure instance
fig, ax = plt.subplots()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    pc.set_facecolor('#344e41')
    pc.set_alpha(1)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_plot[partname]
        vp.set_edgecolor("#1b263b")
        vp.set_linewidth(1)
        


labels = ['Historical','1520-1550','1530-1560','1540-1570','1550-1580']

x1 = [1,2,3,4,5]
ax.set_xticks(x1)
ax.set_xticklabels(labels, minor=False, rotation=45)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yscale('log')
plt.title('Annual Accumulated-South')
plt.savefig('E:/CALFEWS-main/hydrology/driest_periods/annual_south_SI.pdf',bbox_inches='tight')


