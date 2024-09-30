# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:02:20 2024

@author: rg727
"""


#############################################################################
def convert_to_WY(row):
    if row['Date'].month>=10:
        return(pd.datetime(row['Date'].year+1,1,1).year)
    else:
        return(pd.datetime(row['Date'].year,1,1).year)
    

##########################################################################################################################################



shasta_data=pd.read_pickle("E:/CALFEWS-main/hydrology/shasta_Q_1550_1580.pkl")
shasta_data['Date']=shasta_data['index']
shasta_data_paleo=shasta_data[shasta_data['frame']=='df_baseline']


shasta_data_paleo['WaterYear'] = shasta_data_paleo.apply(lambda x: convert_to_WY(x), axis=1)
shasta_data_paleo=shasta_data_paleo.groupby(['ensemble','WaterYear']).sum()
shasta_data_paleo['location']='Shasta'
shasta_data_paleo['frame']='df_baseline'




shasta_data=pd.read_pickle("E:/CALFEWS-main/hydrology/shasta_Q_1550_1580.pkl")
shasta_data['Date']=shasta_data['index']
shasta_data_cc=shasta_data[shasta_data['frame']=='df']


shasta_data_cc['WaterYear'] = shasta_data_cc.apply(lambda x: convert_to_WY(x), axis=1)
shasta_data_cc=shasta_data_cc.groupby(['ensemble','WaterYear']).sum()
shasta_data_cc['location']='Shasta'
shasta_data_cc['frame']='df'








fig, ax = plt.subplots()


df_CC = shasta_data_cc
df_CC=df_CC.reset_index()
df_CC=df_CC.dropna()



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


df_baseline =shasta_data_paleo
df_baseline=df_baseline.reset_index()
df_baseline=df_baseline.dropna()


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



data['x']=data['x']*(1233/(10**6))



ax=sns.boxplot(y='x', x='Duration', data=data, hue='Frame',palette=color_dict,boxprops=dict(alpha=0.6),showfliers=False)

plt.legend([],[], frameon=False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_ylabel('')    
ax.set_xlabel('')


plt.savefig('E:/CALFEWS-main/hydrology/shasta_inflows_accum_SI.pdf',bbox_inches='tight')


###########################################################################################################



#################################################################################


oroville_data=pd.read_pickle("E:/CALFEWS-main/hydrology/oroville_Q_1550_1580.pkl")
oroville_data['Date']=oroville_data['index']
oroville_data_paleo=oroville_data[oroville_data['frame']=='df_baseline']


oroville_data_paleo['WaterYear'] = oroville_data_paleo.apply(lambda x: convert_to_WY(x), axis=1)
oroville_data_paleo=oroville_data_paleo.groupby(['ensemble','WaterYear']).sum()
oroville_data_paleo['location']='oroville'
oroville_data_paleo['frame']='df_baseline'




oroville_data=pd.read_pickle("E:/CALFEWS-main/hydrology/oroville_Q_1550_1580.pkl")
oroville_data['Date']=oroville_data['index']
oroville_data_cc=oroville_data[oroville_data['frame']=='df']


oroville_data_cc['WaterYear'] = oroville_data_cc.apply(lambda x: convert_to_WY(x), axis=1)
oroville_data_cc=oroville_data_cc.groupby(['ensemble','WaterYear']).sum()
oroville_data_cc['location']='oroville'
oroville_data_cc['frame']='df'






fig, ax = plt.subplots()


df_CC = oroville_data_cc
df_CC=df_CC.reset_index()
df_CC=df_CC.dropna()



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


df_baseline =oroville_data_paleo
df_baseline=df_baseline.reset_index()
df_baseline=df_baseline.dropna()


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




data['x']=data['x']*(1233/(10**6))

ax=sns.boxplot(y='x', x='Duration', data=data, hue='Frame',palette=color_dict,boxprops=dict(alpha=0.6),showfliers=False)

plt.legend([],[], frameon=False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_ylabel('')    
ax.set_xlabel('')




plt.savefig('E:/CALFEWS-main/hydrology/oroville_inflows_accum_SI.pdf',bbox_inches='tight')


###################################################################################################
SHA_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/SHA_baseline.txt",sep='\t')
SHA_paleo = SHA_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
SHA_paleo['WaterYear'] = SHA_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)


area_ft2_SHA = 6665*5280*5280

#Cut down to 30 year period 

SHA_paleo_truncated=SHA_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

SHA_paleo_truncated.index=date

SHA_paleo_median_yearly_accum=SHA_paleo_truncated.groupby(['WaterYear']).sum()*area_ft2_SHA/304.8/43560   

SHA_baseline=SHA_paleo_median_yearly_accum.iloc[:,6:56].to_numpy().reshape([1500,1])
SHA_baseline=pd.DataFrame(SHA_baseline)



SHA_baseline['location']='Shasta'
SHA_baseline['frame']='df_baseline'
###############################################################################################################################

SHA_paleo=pd.read_csv("E:/CALFEWS-main/hydrology/SHA_4T_1CC_streamflow.txt",sep='\t')
SHA_paleo = SHA_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#SHA_paleo['Date']=pd.to_datetime(SHA_paleo[['year', 'month', 'day']])
SHA_paleo['WaterYear'] = SHA_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)


area_ft2_SHA = 6665*5280*5280

#Cut down to 30 year period 

SHA_paleo_truncated=SHA_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

SHA_paleo_truncated.index=date
SHA_paleo_median_yearly_accum=SHA_paleo_truncated.groupby(['WaterYear']).sum()

SHA_CC=SHA_paleo_median_yearly_accum.iloc[:,6:56].to_numpy().reshape([1500,1])*area_ft2_SHA/304.8/43560   
SHA_CC=pd.DataFrame(SHA_CC)


SHA_CC['location']='Shasta'
SHA_CC['frame']='df'
###########################################################################################################

ORO_paleo=pd.read_csv("E:/WGEN_Paper_Figures/paleo_datasets/ORO_baseline.txt",sep='\t')
ORO_paleo = ORO_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#ORO_paleo['Date']=pd.to_datetime(ORO_paleo[['year', 'month', 'day']])
ORO_paleo['WaterYear'] = ORO_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)


area_ft2_ORO = 3607*5280*5280

#Cut down to 30 year period 

ORO_paleo_truncated=ORO_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

ORO_paleo_truncated.index=date

ORO_paleo_median_yearly_accum=ORO_paleo_truncated.groupby(['WaterYear']).sum()

ORO_baseline=ORO_paleo_median_yearly_accum.iloc[:,6:56].to_numpy().reshape([1500,1])*area_ft2_ORO/304.8/43560   
ORO_baseline=pd.DataFrame(ORO_baseline)



ORO_baseline['location']='Oroville'
ORO_baseline['frame']='df_baseline'


######################

ORO_paleo=pd.read_csv("E:/CALFEWS-main/hydrology/ORO_4T_1CC_streamflow.txt",sep='\t')
ORO_paleo = ORO_paleo.rename(columns={'sim_datemat_1': 'year', 'sim_datemat_2': 'month','sim_datemat_3': 'day'})
#ORO_paleo['Date']=pd.to_datetime(ORO_paleo[['year', 'month', 'day']])
ORO_paleo['WaterYear'] = ORO_paleo.apply(lambda x: convert_to_WY_paleo(x), axis=1)


area_ft2_ORO = 3607*5280*5280

#Cut down to 30 year period 

ORO_paleo_truncated=ORO_paleo.iloc[55120:66078,:]

date=pd.date_range(start="1730-10-01",end="1760-09-30")

ORO_paleo_truncated.index=date


ORO_paleo_median_yearly_accum=ORO_paleo_truncated.groupby(['WaterYear']).sum()

ORO_CC=ORO_paleo_median_yearly_accum.iloc[:,6:56].to_numpy().reshape([1500,1])*area_ft2_ORO/304.8/43560   
ORO_CC=pd.DataFrame(ORO_CC)



ORO_CC['location']='Oroville'
ORO_CC['frame']='df'


combined_data=pd.concat([SHA_baseline,SHA_CC,ORO_baseline,ORO_CC])



color_dict = dict({'df_baseline':'#283618','df':'#BC6C25'})

combined_data.columns=['Flow','location','frame']

combined_data['Flow']=combined_data['Flow']*(1233/(10**6))

ax=sns.boxplot(y='Flow', x='location', data=combined_data, hue='frame',palette=color_dict,boxprops=dict(alpha=0.6),showfliers=False)
plt.legend([],[], frameon=False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('E:/CALFEWS-main/hydrology/reservoir_fnf_total_SI.pdf',bbox_inches='tight')

#########################################################################

#################################################################################Reservoir Inflows############
oroville_data=pd.read_pickle("E:/CALFEWS-main/hydrology/oroville_Q_1550_1580.pkl")
oroville_data['Date']=oroville_data['index']
oroville_data_paleo=oroville_data[oroville_data['frame']=='df_baseline']


oroville_data_paleo['WaterYear'] = oroville_data_paleo.apply(lambda x: convert_to_WY(x), axis=1)
oroville_data_paleo=oroville_data_paleo.groupby(['ensemble','WaterYear']).sum()
oroville_data_paleo['location']='Oroville'
oroville_data_paleo['frame']='df_baseline'


oroville_data=pd.read_pickle("E:/CALFEWS-main/hydrology/oroville_Q_1550_1580.pkl")
oroville_data['Date']=oroville_data['index']
oroville_data_cc=oroville_data[oroville_data['frame']=='df']


oroville_data_cc['WaterYear'] = oroville_data_cc.apply(lambda x: convert_to_WY(x), axis=1)
oroville_data_cc=oroville_data_cc.groupby(['ensemble','WaterYear']).sum()
oroville_data_cc['location']='Oroville'
oroville_data_cc['frame']='df'




shasta_data=pd.read_pickle("E:/CALFEWS-main/hydrology/shasta_Q_1550_1580.pkl")
shasta_data['Date']=shasta_data['index']
shasta_data_paleo=shasta_data[shasta_data['frame']=='df_baseline']


shasta_data_paleo['WaterYear'] = shasta_data_paleo.apply(lambda x: convert_to_WY(x), axis=1)
shasta_data_paleo=shasta_data_paleo.groupby(['ensemble','WaterYear']).sum()
shasta_data_paleo['location']='Shasta'
shasta_data_paleo['frame']='df_baseline'


shasta_data=pd.read_pickle("E:/CALFEWS-main/hydrology/shasta_Q_1550_1580.pkl")
shasta_data['Date']=shasta_data['index']
shasta_data_cc=shasta_data[shasta_data['frame']=='df']


shasta_data_cc['WaterYear'] = shasta_data_cc.apply(lambda x: convert_to_WY(x), axis=1)
shasta_data_cc=shasta_data_cc.groupby(['ensemble','WaterYear']).sum()
shasta_data_cc['location']='Shasta'
shasta_data_cc['frame']='df'



combined_data=pd.concat([shasta_data_paleo,shasta_data_cc,oroville_data_paleo,oroville_data_cc])




color_dict = dict({'df_baseline':'#283618','df':'#BC6C25'})


combined_data['x']=combined_data['x']*(1233/(10**6))

ax=sns.boxplot(y='x', x='location', data=combined_data, hue='frame',palette=color_dict,boxprops=dict(alpha=0.6),showfliers=False)
plt.legend([],[], frameon=False)
plt.yscale('log')    
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('E:/CALFEWS-main/hydrology/oroville_shasta_inflows_annual_SI.pdf')

