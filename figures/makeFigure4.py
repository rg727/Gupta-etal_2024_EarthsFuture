
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt


import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
	
import matplotlib.patches as mpatches


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


###################################################################################
#SHA paleo 

SHA_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SHA_baseline.txt",sep='\t')
SHA_paleo = SHA_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
SHA_paleo['WaterYear'] = SHA_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_SHA = 6665*5280*5280

SHA_paleo=SHA_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(SHA_paleo.iloc[:,i]*area_ft2_SHA/304.8/43560)
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[:,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

for i in range(1500):
    if df.iloc[i,0]>10**7:
        df.iloc[i,0]=float("NaN")        



data_SHA = df

#ORO paleo

ORO_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ORO_baseline.txt",sep='\t')
ORO_paleo = ORO_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
ORO_paleo['WaterYear'] = ORO_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_ORO = 3607*5280*5280

ORO_paleo=ORO_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(ORO_paleo.iloc[:,i]*area_ft2_ORO/304.8/43560)
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[:,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_ORO = df


#FOL paleo

FOL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/FOL_baseline.txt",sep='\t')
FOL_paleo = FOL_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
FOL_paleo['WaterYear'] = FOL_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_FOL = 1885*5280*5280

FOL_paleo=FOL_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(FOL_paleo.iloc[:,i]*area_ft2_FOL/304.8/43560)
df.columns=['FOL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(FOL_paleo.iloc[:,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_FOL = df

#DNP paleo

DNP_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TLG_baseline.txt",sep='\t')
DNP_paleo = DNP_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
DNP_paleo['WaterYear'] = DNP_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_DNP = 1538*5280*5280

DNP_paleo=DNP_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(DNP_paleo.iloc[:,i]*area_ft2_DNP/304.8/43560)
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[:,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_DNP = df

#NML paleo

NML_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/NML_baseline.txt",sep='\t')
NML_paleo = NML_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
NML_paleo['WaterYear'] = NML_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_NML = 900*5280*5280

NML_paleo=NML_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(NML_paleo.iloc[:,i]*area_ft2_NML/304.8/43560)
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[:,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_NML = df

YRS_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/YRS_baseline.txt",sep='\t')
YRS_paleo = YRS_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
YRS_paleo['WaterYear'] = YRS_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_YRS = 1108*5280*5280

YRS_paleo=YRS_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(YRS_paleo.iloc[:,i]*area_ft2_YRS/304.8/43560)
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[:,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_YRS = df


#EXC paleo

EXC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/EXC_baseline.txt",sep='\t')
EXC_paleo = EXC_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
EXC_paleo['WaterYear'] = EXC_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_EXC = 1061*5280*5280

EXC_paleo=EXC_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(EXC_paleo.iloc[:,i]*area_ft2_EXC/304.8/43560)
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[:,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_EXC = df

paleo_north=data_SHA.iloc[:,0]+data_ORO.iloc[:,0]+data_YRS.iloc[:,0]+data_FOL.iloc[:,0]+data_EXC.iloc[:,0]+data_NML.iloc[:,0]+data_DNP.iloc[:,0]

paleo_north=pd.DataFrame(paleo_north)
paleo_north['Year']=years_array
paleo_north.columns=['Streamflow','Year']

paleo_north['Year']=paleo_north['Year'].rolling(15).mean()
paleo_north['Streamflow']=paleo_north['Streamflow'].rolling(15).mean()


paleo_north['Streamflow']=paleo_north['Streamflow']*1233
# Create a figure instance
fig,ax = plt.subplots()
sns.lineplot(data=paleo_north, x="Year", y="Streamflow",color='#432818')

left, bottom, width, height = (1520,40000*1233,60,70000*1233)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=True,
                       facecolor="grey",alpha=0.4)
plt.gca().add_patch(rect)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
from matplotlib import ticker
M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.xaxis.set_major_locator(xticks)

plt.savefig('E:/CALFEWS-main/hydrology/north_time_series_SI.pdf',bbox_inches='tight')


#################################################################Southern Hydrology########################################################

#MIL paleo

MIL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/MIL_baseline.txt",sep='\t')
MIL_paleo = MIL_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
MIL_paleo['WaterYear'] = MIL_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_MIL = 1675*5280*5280

MIL_paleo=MIL_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(MIL_paleo.iloc[:,i]*area_ft2_MIL/304.8/43560)
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[:,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_MIL = df


#ISB paleo

ISB_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ISB_baseline.txt",sep='\t')
ISB_paleo = ISB_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
ISB_paleo['WaterYear'] = ISB_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_ISB = 2074*5280*5280

ISB_paleo=ISB_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(ISB_paleo.iloc[:,i]*area_ft2_ISB/304.8/43560)
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[:,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_ISB = df


#SUC paleo

SUC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SUC_baseline.txt",sep='\t')
SUC_paleo = SUC_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
SUC_paleo['WaterYear'] = SUC_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_SUC = 393*5280*5280

SUC_paleo=SUC_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(SUC_paleo.iloc[:,i]*area_ft2_SUC/304.8/43560)
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[:,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_SUC = df


#KWH paleo

KWH_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TRM_baseline.txt",sep='\t')
KWH_paleo = KWH_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
KWH_paleo['WaterYear'] = KWH_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_KWH = 561*5280*5280

KWH_paleo=KWH_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(KWH_paleo.iloc[:,i]*area_ft2_KWH/304.8/43560)
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[:,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_KWH = df


#PNF paleo

PNF_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/PNF_baseline.txt",sep='\t')
PNF_paleo = PNF_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
PNF_paleo['WaterYear'] = PNF_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_PNF = 1545*5280*5280

PNF_paleo=PNF_paleo.groupby(['WaterYear']).mean()

i=6
df=pd.DataFrame(PNF_paleo.iloc[:,i]*area_ft2_PNF/304.8/43560)
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[:,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_PNF = df





paleo_south=data_MIL.iloc[:,0]+data_SUC.iloc[:,0]+data_ISB.iloc[:,0]+data_KWH.iloc[:,0]+data_PNF.iloc[:,0]


paleo_south=pd.DataFrame(paleo_south)
paleo_south['Year']=years_array
paleo_south.columns=['Streamflow','Year']

paleo_south['Year']=paleo_south['Year'].rolling(15).mean()
paleo_south['Streamflow']=paleo_south['Streamflow'].rolling(15).mean()


paleo_south['Streamflow']=paleo_south['Streamflow']*1233

fig,ax = plt.subplots()
sns.lineplot(data=paleo_south, x="Year", y="Streamflow",color='#432818')
left, bottom, width, height = (1520,7000*1233, 60,20000*1233)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=True,
                       facecolor="grey",alpha=0.4)
plt.gca().add_patch(rect)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
from matplotlib import ticker
M = 4
yticks = ticker.MaxNLocator(5)
xticks = ticker.MaxNLocator(M)
ax.xaxis.set_major_locator(xticks)      
ax.yaxis.set_major_locator(yticks)     
        


plt.savefig('E:/CALFEWS-main/hydrology/south_time_series_SI.pdf',bbox_inches='tight')


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
#SHA_paleo = SHA_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
SHA_paleo_truncated=SHA_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

SHA_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_SHA = 6665*5280*5280


i=6
df=pd.DataFrame(SHA_paleo_truncated.iloc[:,i]*area_ft2_SHA/304.8/43560)
df=df.resample('M').min()
df.columns=['SHA_fnf']




for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo_truncated.iloc[:,i]*area_ft2_SHA/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_SHA = df
######################################################################################################

ORO_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ORO_baseline.txt",sep='\t')
#ORO_paleo = ORO_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
ORO_paleo_truncated=ORO_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

ORO_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_ORO = 3607*5280*5280


i=6
df=pd.DataFrame(ORO_paleo_truncated.iloc[:,i]*area_ft2_ORO/304.8/43560)
df=df.resample('M').min()
df.columns=['ORO_fnf']




for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo_truncated.iloc[:,i]*area_ft2_ORO/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_ORO = df

############################################

FOL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/FOL_baseline.txt",sep='\t')
#FOL_paleo = FOL_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
FOL_paleo_truncated=FOL_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

FOL_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_FOL = 1885*5280*5280


i=6
df=pd.DataFrame(FOL_paleo_truncated.iloc[:,i]*area_ft2_FOL/304.8/43560)
df=df.resample('M').min()
df.columns=['FOL_fnf']




for i in range(7,56):
    df1=pd.DataFrame(FOL_paleo_truncated.iloc[:,i]*area_ft2_FOL/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_FOL = df

####################################################DNP paleo


DNP_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TLG_baseline.txt",sep='\t')
#DNP_paleo = DNP_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
DNP_paleo_truncated=DNP_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

DNP_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_DNP = 1538*5280*5280


i=6
df=pd.DataFrame(DNP_paleo_truncated.iloc[:,i]*area_ft2_DNP/304.8/43560)
df=df.resample('M').min()
df.columns=['DNP_fnf']




for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo_truncated.iloc[:,i]*area_ft2_DNP/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_DNP = df

##############################################
NML_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/NML_baseline.txt",sep='\t')
#NML_paleo = NML_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
NML_paleo_truncated=NML_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

NML_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_NML = 900*5280*5280


i=6
df=pd.DataFrame(NML_paleo_truncated.iloc[:,i]*area_ft2_NML/304.8/43560)
df=df.resample('M').min()
df.columns=['NML_fnf']




for i in range(7,56):
    df1=pd.DataFrame(NML_paleo_truncated.iloc[:,i]*area_ft2_NML/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_NML = df
###################################################

YRS_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/YRS_baseline.txt",sep='\t')
#YRS_paleo = YRS_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
YRS_paleo_truncated=YRS_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

YRS_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_YRS = 1108*5280*5280


i=6
df=pd.DataFrame(YRS_paleo_truncated.iloc[:,i]*area_ft2_YRS/304.8/43560)
df=df.resample('M').min()
df.columns=['YRS_fnf']




for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo_truncated.iloc[:,i]*area_ft2_YRS/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_YRS = df


EXC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/EXC_baseline.txt",sep='\t')
#EXC_paleo = EXC_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
EXC_paleo_truncated=EXC_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

EXC_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_EXC = 1061*5280*5280


i=6
df=pd.DataFrame(EXC_paleo_truncated.iloc[:,i]*area_ft2_EXC/304.8/43560)
df=df.resample('M').min()
df.columns=['EXC_fnf']




for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo_truncated.iloc[:,i]*area_ft2_EXC/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_EXC = df

paleo_north=data_SHA.iloc[:,0]+data_ORO.iloc[:,0]+data_YRS.iloc[:,0]+data_FOL.iloc[:,0]+data_EXC.iloc[:,0]+data_NML.iloc[:,0]+data_DNP.iloc[:,0]

#########################################################################
MIL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/MIL_baseline.txt",sep='\t')
#MIL_paleo = MIL_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
MIL_paleo_truncated=MIL_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

MIL_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_MIL = 1675*5280*5280


i=6
df=pd.DataFrame(MIL_paleo_truncated.iloc[:,i]*area_ft2_MIL/304.8/43560)
df=df.resample('M').min()
df.columns=['MIL_fnf']




for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo_truncated.iloc[:,i]*area_ft2_MIL/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_MIL = df


ISB_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ISB_baseline.txt",sep='\t')
#ISB_paleo = ISB_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
ISB_paleo_truncated=ISB_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

ISB_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_ISB = 2074*5280*5280


i=6
df=pd.DataFrame(ISB_paleo_truncated.iloc[:,i]*area_ft2_ISB/304.8/43560)
df=df.resample('M').min()
df.columns=['ISB_fnf']




for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo_truncated.iloc[:,i]*area_ft2_ISB/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_ISB = df


SUC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SUC_baseline.txt",sep='\t')
#SUC_paleo = SUC_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
SUC_paleo_truncated=SUC_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

SUC_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_SUC = 393*5280*5280


i=6
df=pd.DataFrame(SUC_paleo_truncated.iloc[:,i]*area_ft2_SUC/304.8/43560)
df=df.resample('M').min()
df.columns=['SUC_fnf']




for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo_truncated.iloc[:,i]*area_ft2_SUC/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['SUC_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_SUC = df




KWH_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TRM_baseline.txt",sep='\t')
#KWH_paleo = KWH_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
KWH_paleo_truncated=KWH_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

KWH_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_KWH = 561*5280*5280


i=6
df=pd.DataFrame(KWH_paleo_truncated.iloc[:,i]*area_ft2_KWH/304.8/43560)
df=df.resample('M').min()
df.columns=['KWH_fnf']




for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo_truncated.iloc[:,i]*area_ft2_KWH/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_KWH = df



PNF_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/PNF_baseline.txt",sep='\t')
#PNF_paleo = PNF_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
PNF_paleo_truncated=PNF_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

PNF_paleo_truncated.index=date

#Aggregate to accumulated WY flow

area_ft2_PNF = 1545*5280*5280


i=6
df=pd.DataFrame(PNF_paleo_truncated.iloc[:,i]*area_ft2_PNF/304.8/43560)
df=df.resample('M').min()
df.columns=['PNF_fnf']




for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo_truncated.iloc[:,i]*area_ft2_PNF/304.8/43560)
    df1=df1.resample('M').min()
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_PNF = df

paleo_south=data_MIL.iloc[:,0]+data_SUC.iloc[:,0]+data_ISB.iloc[:,0]+data_KWH.iloc[:,0]+data_PNF.iloc[:,0]



## combine these different collections into a list
data_to_plot = [np.array(CDEC_south.iloc[:,0])*1233,np.array(paleo_south)*1233]



# Create a figure instance
fig = plt.figure()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    if i==0:
        pc.set_facecolor('#97D8C4')
        pc.set_alpha(1)        
        for partname in ('cbars', 'cmins', 'cmaxes'):
            vp = violin_plot[partname]
            vp.set_edgecolor("#1b263b")
            vp.set_linewidth(1)
    else:
        pc.set_facecolor('#344e41')
        pc.set_alpha(1)        
        
plt.yscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off


plt.savefig('E:/CALFEWS-main/hydrology/monthly_min_south_SI.pdf',bbox_inches='tight')




## combine these different collections into a list
data_to_plot = [CDEC_north.iloc[:,0]*1233,paleo_north*1233]


# Create a figure instance
fig = plt.figure()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    if i==0:
        pc.set_facecolor('#97D8C4')
        pc.set_alpha(1)        
        for partname in ('cbars', 'cmins', 'cmaxes'):
            vp = violin_plot[partname]
            vp.set_edgecolor("#1b263b")
            vp.set_linewidth(1)
    else:
        pc.set_facecolor('#344e41')
        pc.set_alpha(1)        
        
plt.yscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off




plt.savefig('E:/CALFEWS-main/hydrology/monthly_min_north_SI.pdf',bbox_inches='tight')


###############################################################################################

#Calculate 3 year, 5-year, 10-year mins 

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
df=pd.DataFrame(SHA_paleo.iloc[55120:66078,i]*area_ft2_SHA/304.8/43560)
df
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[55120:66078,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)

data_SHA = df



#ORO paleo

ORO_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ORO_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_ORO = 3607*5280*5280


i=6
df=pd.DataFrame(ORO_paleo.iloc[55120:66078,i]*area_ft2_ORO/304.8/43560)
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[55120:66078,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_ORO = df


#FOL paleo

FOL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/FOL_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_FOL = 1885*5280*5280


i=6
df=pd.DataFrame(FOL_paleo.iloc[55120:66078,i]*area_ft2_FOL/304.8/43560)
df.columns=['FOL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(FOL_paleo.iloc[55120:66078,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_FOL = df

#DNP paleo

DNP_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TLG_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_DNP = 1538*5280*5280


i=6
df=pd.DataFrame(DNP_paleo.iloc[55120:66078,i]*area_ft2_DNP/304.8/43560)
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[55120:66078,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_DNP = df

#NML paleo

NML_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/NML_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_NML = 900*5280*5280


i=6
df=pd.DataFrame(NML_paleo.iloc[55120:66078,i]*area_ft2_NML/304.8/43560)
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[55120:66078,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_NML = df

YRS_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/YRS_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_YRS = 1108*5280*5280


i=6
df=pd.DataFrame(YRS_paleo.iloc[55120:66078,i]*area_ft2_YRS/304.8/43560)
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[55120:66078,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_YRS = df


#EXC paleo

EXC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/EXC_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_EXC = 1061*5280*5280


i=6
df=pd.DataFrame(EXC_paleo.iloc[55120:66078,i]*area_ft2_EXC/304.8/43560)
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[55120:66078,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_EXC = df

paleo_north=data_SHA.iloc[:,0]+data_ORO.iloc[:,0]+data_YRS.iloc[:,0]+data_FOL.iloc[:,0]+data_EXC.iloc[:,0]+data_NML.iloc[:,0]+data_DNP.iloc[:,0]


#MIL paleo

MIL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/MIL_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_MIL = 1675*5280*5280


i=6
df=pd.DataFrame(MIL_paleo.iloc[55120:66078,i]*area_ft2_MIL/304.8/43560)
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[55120:66078,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_MIL = df


#ISB paleo

ISB_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ISB_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_ISB = 2074*5280*5280

i=6
df=pd.DataFrame(ISB_paleo.iloc[55120:66078,i]*area_ft2_ISB/304.8/43560)
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[55120:66078,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_ISB = df


#SUC paleo

SUC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SUC_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_SUC = 393*5280*5280


i=6
df=pd.DataFrame(SUC_paleo.iloc[55120:66078,i]*area_ft2_SUC/304.8/43560)
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[55120:66078,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_SUC = df


#KWH paleo

KWH_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TRM_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_KWH = 561*5280*5280


i=6
df=pd.DataFrame(KWH_paleo.iloc[55120:66078,i]*area_ft2_KWH/304.8/43560)
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[55120:66078,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_KWH = df


#PNF paleo

PNF_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/PNF_baseline.txt",sep='\t')

#Aggregate to accumulated WY flow

area_ft2_PNF = 1545*5280*5280


i=6
df=pd.DataFrame(PNF_paleo.iloc[55120:66078,i]*area_ft2_PNF/304.8/43560)
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[55120:66078,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_PNF = df



paleo_south=data_MIL.iloc[:,0]+data_SUC.iloc[:,0]+data_ISB.iloc[:,0]+data_KWH.iloc[:,0]+data_PNF.iloc[:,0]


##########################################calculate rolling min####################################
n=30*365

paleo_north_three_rolling=pd.DataFrame(paleo_north.rolling(3*365).mean())
paleo_north_three=paleo_north_three_rolling.groupby(np.arange(len(paleo_north_three_rolling))//n).min()

paleo_north_five_rolling=pd.DataFrame(paleo_north.rolling(5*365).mean())
paleo_north_five=paleo_north_five_rolling.groupby(np.arange(len(paleo_north_five_rolling))//n).min()

paleo_north_ten_rolling=pd.DataFrame(paleo_north.rolling(10*365).mean())
paleo_north_ten=paleo_north_ten_rolling.groupby(np.arange(len(paleo_north_ten_rolling))//n).min()



paleo_south_three_rolling=pd.DataFrame(paleo_south.rolling(3*365).mean())
paleo_south_three=paleo_south_three_rolling.groupby(np.arange(len(paleo_south_three_rolling))//n).min()

paleo_south_five_rolling=pd.DataFrame(paleo_south.rolling(5*365).mean())
paleo_south_five=paleo_south_five_rolling.groupby(np.arange(len(paleo_south_five_rolling))//n).min()

paleo_south_ten_rolling=pd.DataFrame(paleo_south.rolling(10*365).mean())
paleo_south_ten=paleo_south_ten_rolling.groupby(np.arange(len(paleo_south_ten_rolling))//n).min()




## combine these different collections into a list
data_to_plot = [np.array(paleo_north_three.iloc[:,0]*1233),np.array(paleo_north_five.iloc[:,0]*1233),np.array(paleo_north_ten.iloc[:,0]*1233)]

data_to_plot=data_to_plot
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
        

ax.plot(1, CDEC_north_daily_three*1233, marker='o' ,color='#97D8C4',markersize=10) 
ax.plot(2, CDEC_north_daily_five*1233, marker='o', color='#97D8C4',markersize=10) 
ax.plot(3, CDEC_north_daily_ten*1233, marker='o',color='#97D8C4',markersize=10 ) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.savefig('E:/CALFEWS-main/hydrology/yearlyminflow_north_SI.pdf',bbox_inches='tight')



## combine these different collections into a list
data_to_plot = [np.array(paleo_south_three.iloc[:,0]*1233),np.array(paleo_south_five.iloc[:,0]*1233),np.array(paleo_south_ten.iloc[:,0]*1233)]
data_to_plot=data_to_plot
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
        

ax.plot(1, CDEC_south_daily_three*1233, marker='o' ,color='#97D8C4',markersize=10) 
ax.plot(2, CDEC_south_daily_five*1233, marker='o', color='#97D8C4',markersize=10) 
ax.plot(3, CDEC_south_daily_ten*1233, marker='o',color='#97D8C4',markersize=10) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig('E:/CALFEWS-main/hydrology/yearlyminflow_south_SI.pdf',bbox_inches='tight')

#############################
###################################################################################
#SHA paleo 

SHA_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SHA_baseline.txt",sep='\t')
SHA_paleo = SHA_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
SHA_paleo['WaterYear'] = SHA_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_SHA = 6665*5280*5280

SHA_paleo=SHA_paleo.groupby(['WaterYear']).sum()
SHA_paleo=SHA_paleo.reset_index()

i=6
df=pd.DataFrame(SHA_paleo.iloc[150:180,i]*area_ft2_SHA/304.8/43560)
df.columns=['SHA_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SHA_paleo.iloc[150:180,i]*area_ft2_SHA/304.8/43560)
    df1.columns=['SHA_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    



data_SHA = df

#ORO paleo

ORO_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ORO_baseline.txt",sep='\t')
ORO_paleo = ORO_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
ORO_paleo['WaterYear'] = ORO_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_ORO = 3607*5280*5280

ORO_paleo=ORO_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(ORO_paleo.iloc[150:180,i]*area_ft2_ORO/304.8/43560)
df.columns=['ORO_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ORO_paleo.iloc[150:180,i]*area_ft2_ORO/304.8/43560)
    df1.columns=['ORO_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_ORO = df


#FOL paleo

FOL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/FOL_baseline.txt",sep='\t')
FOL_paleo = FOL_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
FOL_paleo['WaterYear'] = FOL_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_FOL = 1885*5280*5280

FOL_paleo=FOL_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(FOL_paleo.iloc[150:180,i]*area_ft2_FOL/304.8/43560)
df.columns=['FOL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(FOL_paleo.iloc[150:180,i]*area_ft2_FOL/304.8/43560)
    df1.columns=['FOL_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_FOL = df

#DNP paleo

DNP_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TLG_baseline.txt",sep='\t')
DNP_paleo = DNP_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
DNP_paleo['WaterYear'] = DNP_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_DNP = 1538*5280*5280

DNP_paleo=DNP_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(DNP_paleo.iloc[150:180,i]*area_ft2_DNP/304.8/43560)
df.columns=['DNP_fnf']



for i in range(7,56):
    df1=pd.DataFrame(DNP_paleo.iloc[150:180,i]*area_ft2_DNP/304.8/43560)
    df1.columns=['DNP_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_DNP = df

#NML paleo

NML_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/NML_baseline.txt",sep='\t')
NML_paleo = NML_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
NML_paleo['WaterYear'] = NML_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_NML = 900*5280*5280

NML_paleo=NML_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(NML_paleo.iloc[150:180,i]*area_ft2_NML/304.8/43560)
df.columns=['NML_fnf']



for i in range(7,56):
    df1=pd.DataFrame(NML_paleo.iloc[150:180,i]*area_ft2_NML/304.8/43560)
    df1.columns=['NML_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_NML = df

YRS_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/YRS_baseline.txt",sep='\t')
YRS_paleo = YRS_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
YRS_paleo['WaterYear'] = YRS_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_YRS = 1108*5280*5280

YRS_paleo=YRS_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(YRS_paleo.iloc[150:180,i]*area_ft2_YRS/304.8/43560)
df.columns=['YRS_fnf']



for i in range(7,56):
    df1=pd.DataFrame(YRS_paleo.iloc[150:180,i]*area_ft2_YRS/304.8/43560)
    df1.columns=['YRS_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_YRS = df


#EXC paleo

EXC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/EXC_baseline.txt",sep='\t')
EXC_paleo = EXC_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
EXC_paleo['WaterYear'] = EXC_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_EXC = 1061*5280*5280

EXC_paleo=EXC_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(EXC_paleo.iloc[150:180,i]*area_ft2_EXC/304.8/43560)
df.columns=['EXC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(EXC_paleo.iloc[150:180,i]*area_ft2_EXC/304.8/43560)
    df1.columns=['EXC_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_EXC = df

paleo_north=data_SHA.iloc[:,0]+data_ORO.iloc[:,0]+data_YRS.iloc[:,0]+data_FOL.iloc[:,0]+data_EXC.iloc[:,0]+data_NML.iloc[:,0]+data_DNP.iloc[:,0]


## combine these different collections into a list
data_to_plot = [CDEC_north.iloc[:,0]*1233,paleo_north*1233]



# Create a figure instance
fig = plt.figure()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    if i==0:
        pc.set_facecolor('#97D8C4')
        pc.set_alpha(1)        
        for partname in ('cbars', 'cmins', 'cmaxes'):
            vp = violin_plot[partname]
            vp.set_edgecolor("#1b263b")
            vp.set_linewidth(1)
    else:
        pc.set_facecolor('#344e41')
        pc.set_alpha(1)        
        
plt.yscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)



plt.savefig('E:/CALFEWS-main/hydrology/accumulated_annual_flow_north_new_1550_SI.pdf',bbox_inches='tight')

#################################################################Southern Hydrology########################################################

#MIL paleo

MIL_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/MIL_baseline.txt",sep='\t')
MIL_paleo = MIL_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
MIL_paleo['WaterYear'] = MIL_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_MIL = 1675*5280*5280

MIL_paleo=MIL_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(MIL_paleo.iloc[150:180,i]*area_ft2_MIL/304.8/43560)
df.columns=['MIL_fnf']



for i in range(7,56):
    df1=pd.DataFrame(MIL_paleo.iloc[150:180,i]*area_ft2_MIL/304.8/43560)
    df1.columns=['MIL_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_MIL = df


#ISB paleo

ISB_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ISB_baseline.txt",sep='\t')
ISB_paleo = ISB_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
ISB_paleo['WaterYear'] = ISB_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_ISB = 2074*5280*5280

ISB_paleo=ISB_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(ISB_paleo.iloc[150:180,i]*area_ft2_ISB/304.8/43560)
df.columns=['ISB_fnf']



for i in range(7,56):
    df1=pd.DataFrame(ISB_paleo.iloc[150:180,i]*area_ft2_ISB/304.8/43560)
    df1.columns=['ISB_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_ISB = df


#SUC paleo

SUC_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SUC_baseline.txt",sep='\t')
SUC_paleo = SUC_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
SUC_paleo['WaterYear'] = SUC_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_SUC = 393*5280*5280

SUC_paleo=SUC_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(SUC_paleo.iloc[150:180,i]*area_ft2_SUC/304.8/43560)
df.columns=['SUC_fnf']



for i in range(7,56):
    df1=pd.DataFrame(SUC_paleo.iloc[150:180,i]*area_ft2_SUC/304.8/43560)
    df1.columns=['SUC_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_SUC = df


#KWH paleo

KWH_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/TRM_baseline.txt",sep='\t')
KWH_paleo = KWH_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
KWH_paleo['WaterYear'] = KWH_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_KWH = 561*5280*5280

KWH_paleo=KWH_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(KWH_paleo.iloc[150:180,i]*area_ft2_KWH/304.8/43560)
df.columns=['KWH_fnf']



for i in range(7,56):
    df1=pd.DataFrame(KWH_paleo.iloc[150:180,i]*area_ft2_KWH/304.8/43560)
    df1.columns=['KWH_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_KWH = df


#PNF paleo

PNF_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/PNF_baseline.txt",sep='\t')
PNF_paleo = PNF_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
PNF_paleo['WaterYear'] = PNF_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)

#Aggregate to accumulated WY flow

area_ft2_PNF = 1545*5280*5280

PNF_paleo=PNF_paleo.groupby(['WaterYear']).sum()

i=6
df=pd.DataFrame(PNF_paleo.iloc[150:180,i]*area_ft2_PNF/304.8/43560)
df.columns=['PNF_fnf']



for i in range(7,56):
    df1=pd.DataFrame(PNF_paleo.iloc[150:180,i]*area_ft2_PNF/304.8/43560)
    df1.columns=['PNF_fnf']
    df=pd.concat([df, df1],ignore_index=True)
    

data_PNF = df





paleo_south=data_MIL.iloc[:,0]+data_SUC.iloc[:,0]+data_ISB.iloc[:,0]+data_KWH.iloc[:,0]+data_PNF.iloc[:,0]


## combine these different collections into a list
data_to_plot = [np.array(CDEC_south.iloc[:,0])*1233,paleo_south*1233]

# Create a figure instance
fig = plt.figure()

# Create the boxplot
violin_plot = plt.violinplot(data_to_plot)

for i, pc in enumerate(violin_plot["bodies"]):
    if i==0:
        pc.set_facecolor('#97D8C4')
        pc.set_alpha(1)        
        for partname in ('cbars', 'cmins', 'cmaxes'):
            vp = violin_plot[partname]
            vp.set_edgecolor("#1b263b")
            vp.set_linewidth(1)
    else:
        pc.set_facecolor('#344e41')
        pc.set_alpha(1)        
        
plt.yscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)



plt.savefig('E:/CALFEWS-main/hydrology/accumulated_annual_flow_south_1550_SI.pdf',bbox_inches='tight')
