# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:08:30 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:25:56 2024

@author: rg727
"""


#Calculate minimum deliveries across all ensemble members 



baseline=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_baseline.csv", sep='\t')
scenario_1T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_1T_0.5CC.csv", sep='\t')
scenario_1T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_1T_0.75CC.csv", sep='\t')
scenario_1T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_1T_1CC.csv", sep='\t')
scenario_2T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_2T_0.5CC.csv", sep='\t')
scenario_2T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_2T_0.75CC.csv", sep='\t')
scenario_2T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_2T_1CC.csv", sep='\t')
scenario_3T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_3T_0.5CC.csv", sep='\t')
scenario_3T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_3T_0.75CC.csv", sep='\t')
scenario_3T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_3T_1CC.csv", sep='\t')
scenario_4T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_4T_0.5CC.csv", sep='\t')
scenario_4T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_4T_0.75CC.csv", sep='\t')
scenario_4T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_4T_1CC.csv", sep='\t')



baseline=baseline.dropna()
#baseline_modern=modern_baseline.dropna()
scenario_1T_05CC=scenario_1T_05CC.dropna()
scenario_1T_075CC=scenario_1T_075CC.dropna()
scenario_1T_1CC=scenario_1T_1CC.dropna()
scenario_2T_05CC=scenario_2T_05CC.dropna()
scenario_2T_075CC=scenario_2T_075CC.dropna()
scenario_2T_1CC=scenario_2T_1CC.dropna()
scenario_3T_05CC=scenario_3T_05CC.dropna()
scenario_3T_075CC=scenario_3T_075CC.dropna()
scenario_3T_1CC=scenario_3T_1CC.dropna()
scenario_4T_05CC=scenario_4T_05CC.dropna()
scenario_4T_075CC=scenario_4T_075CC.dropna()
scenario_4T_1CC=scenario_4T_1CC.dropna()




#Create a 5 year rolling average deliveries 
baseline_swpdelta=np.nanmin(baseline['x'].rolling(window=5).sum())
threshold_1T_05CC=np.nanmin(scenario_1T_05CC['x'].rolling(window=5).sum())
threshold_1T_075CC=np.nanmin(scenario_1T_075CC['x'].rolling(window=5).sum())
threshold_1T_1CC=np.nanmin(scenario_1T_1CC['x'].rolling(window=5).sum())

threshold_2T_05CC=np.nanmin(scenario_2T_05CC['x'].rolling(window=5).sum())
threshold_2T_075CC=np.nanmin(scenario_2T_075CC['x'].rolling(window=5).sum())
threshold_2T_1CC=np.nanmin(scenario_2T_1CC['x'].rolling(window=5).sum())

threshold_3T_05CC=np.nanmin(scenario_3T_05CC['x'].rolling(window=5).sum())
threshold_3T_075CC=np.nanmin(scenario_3T_075CC['x'].rolling(window=5).sum())
threshold_3T_1CC=np.nanmin(scenario_3T_1CC['x'].rolling(window=5).sum())

threshold_4T_05CC=np.nanmin(scenario_4T_05CC['x'].rolling(window=5).sum())
threshold_4T_075CC=np.nanmin(scenario_4T_075CC['x'].rolling(window=5).sum())
threshold_4T_1CC=np.nanmin(scenario_4T_1CC['x'].rolling(window=5).sum())



df = pd.DataFrame(columns=['0.5CC', '0.75CC', '1CC','min','max'], index=range(4))

df.iloc[:,0]=np.array([threshold_1T_05CC,threshold_2T_05CC,threshold_3T_05CC,threshold_4T_05CC])
df.iloc[:,1]=np.array([threshold_1T_075CC,threshold_2T_075CC,threshold_3T_075CC,threshold_4T_075CC])
df.iloc[:,2]=np.array([threshold_1T_1CC,threshold_2T_1CC,threshold_3T_1CC,threshold_4T_1CC])

df.iloc[:,3]=df.min(axis = 1)
df.iloc[:,4]=df.max(axis = 1)




datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_truncated=datDaily[datDaily["Date"].dt.month.isin([9])]

datDaily_monthly=datDaily_truncated.resample('M').mean()

test_CDEC=datDaily_monthly['tableA_contract']
test_CDEC=test_CDEC.dropna()
test_CDEC=pd.DataFrame(test_CDEC)

#Create a 5 year rolling average deliveries 
test_CDEC_5_swp=np.nanmin(test_CDEC.rolling(window=5).sum())





#####################################################################################
#Read in scenarios 
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:54:55 2024

@author: rg727
"""


baseline=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_baseline.csv", sep='\t')
scenario_1T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_1T_0.5CC.csv", sep='\t')
scenario_1T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_1T_0.75CC.csv", sep='\t')
scenario_1T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_1T_1CC.csv", sep='\t')
scenario_2T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_2T_0.5CC.csv", sep='\t')
scenario_2T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_2T_0.75CC.csv", sep='\t')
scenario_2T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_2T_1CC.csv", sep='\t')
scenario_3T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_3T_0.5CC.csv", sep='\t')
scenario_3T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_3T_0.75CC.csv", sep='\t')
scenario_3T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_3T_1CC.csv", sep='\t')
scenario_4T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_4T_0.5CC.csv", sep='\t')
scenario_4T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_4T_0.75CC.csv", sep='\t')
scenario_4T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/cvpdelta_contract_4T_1CC.csv", sep='\t')



baseline=baseline.dropna()
scenario_1T_05CC=scenario_1T_05CC.dropna()
scenario_1T_075CC=scenario_1T_075CC.dropna()
scenario_1T_1CC=scenario_1T_1CC.dropna()
scenario_2T_05CC=scenario_2T_05CC.dropna()
scenario_2T_075CC=scenario_2T_075CC.dropna()
scenario_2T_1CC=scenario_2T_1CC.dropna()
scenario_3T_05CC=scenario_3T_05CC.dropna()
scenario_3T_075CC=scenario_3T_075CC.dropna()
scenario_3T_1CC=scenario_3T_1CC.dropna()
scenario_4T_05CC=scenario_4T_05CC.dropna()
scenario_4T_075CC=scenario_4T_075CC.dropna()
scenario_4T_1CC=scenario_4T_1CC.dropna()




#Create a 5 year rolling average deliveries 
baseline_cvpdelta=np.nanmin(baseline['x'].rolling(window=5).sum())
threshold_1T_05CC=np.nanmin(scenario_1T_05CC['x'].rolling(window=5).sum())
threshold_1T_075CC=np.nanmin(scenario_1T_075CC['x'].rolling(window=5).sum())
threshold_1T_1CC=np.nanmin(scenario_1T_1CC['x'].rolling(window=5).sum())

threshold_2T_05CC=np.nanmin(scenario_2T_05CC['x'].rolling(window=5).sum())
threshold_2T_075CC=np.nanmin(scenario_2T_075CC['x'].rolling(window=5).sum())
threshold_2T_1CC=np.nanmin(scenario_2T_1CC['x'].rolling(window=5).sum())

threshold_3T_05CC=np.nanmin(scenario_3T_05CC['x'].rolling(window=5).sum())
threshold_3T_075CC=np.nanmin(scenario_3T_075CC['x'].rolling(window=5).sum())
threshold_3T_1CC=np.nanmin(scenario_3T_1CC['x'].rolling(window=5).sum())

threshold_4T_05CC=np.nanmin(scenario_4T_05CC['x'].rolling(window=5).sum())
threshold_4T_075CC=np.nanmin(scenario_4T_075CC['x'].rolling(window=5).sum())
threshold_4T_1CC=np.nanmin(scenario_4T_1CC['x'].rolling(window=5).sum())




df_cvpdelta = pd.DataFrame(columns=['0.5CC', '0.75CC', '1CC','min','max'], index=range(4))

df_cvpdelta.iloc[:,0]=np.array([threshold_1T_05CC,threshold_2T_05CC,threshold_3T_05CC,threshold_4T_05CC])
df_cvpdelta.iloc[:,1]=np.array([threshold_1T_075CC,threshold_2T_075CC,threshold_3T_075CC,threshold_4T_075CC])
df_cvpdelta.iloc[:,2]=np.array([threshold_1T_1CC,threshold_2T_1CC,threshold_3T_1CC,threshold_4T_1CC])
df_cvpdelta.iloc[:,3]=df_cvpdelta.min(axis = 1)
df_cvpdelta.iloc[:,4]=df_cvpdelta.max(axis = 1)



datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_truncated=datDaily[datDaily["Date"].dt.month.isin([9])]

datDaily_monthly=datDaily_truncated.resample('M').mean()

test_CDEC=datDaily_monthly['cvpdelta_contract']
test_CDEC=test_CDEC.dropna()
test_CDEC=pd.DataFrame(test_CDEC)

#Create a 5 year rolling average deliveries 
test_CDEC_5_cvp=np.nanmin(test_CDEC.rolling(window=5).sum())






fig = plt.figure()
ax = plt.axes()
#ax.plot(range(1,5),df.iloc[:,0],color="#ee9b00")
#ax.plot(range(1,5),df.iloc[:,1],color="#ca6702")
#ax.plot(range(1,5),df.iloc[:,2],color="#ae2012")
ax.axhline(test_CDEC_5_cvp*1233/(10**6), linestyle='--', color='#0d1b2a', linewidth=2,alpha=0.6)
ax.axhline(test_CDEC_5_swp*1233/(10**6), linestyle='--', color='#778da9', linewidth=2,alpha=0.6)
ax.fill_between(range(1,5),df.iloc[:,3]*1233/(10**6),df.iloc[:,4]*1233/(10**6),color="#778da9",alpha=0.6)
#ax.plot(range(1,5),df_cvpdelta.iloc[:,0],color="#0a9396")
#ax.plot(range(1,5),df_cvpdelta.iloc[:,1],color="#005f73")
#ax.plot(range(1,5),df_cvpdelta.iloc[:,2],color="#001219")
ax.fill_between(range(1,5),df_cvpdelta.iloc[:,3]*1233/(10**6),df_cvpdelta.iloc[:,4]*1233/(10**6),color="#0d1b2a",alpha=0.6)
plt.plot(0,baseline_cvpdelta*1233/(10**6),'o',color="#0d1b2a",markersize=10) 
plt.plot(0,baseline_swpdelta*1233/(10**6),'o',color="#778da9",markersize=10) 
plt.xticks(range(0,5))
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('E:/CALFEWS-main/cc_change_heatmap/swpdelta_contract_cvpdelta_new_SI.pdf',bbox_inches='tight')



#############################################################################################################

baseline=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_baseline.csv", sep='\t')
scenario_1T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_1T_0.5CC.csv", sep='\t')
scenario_1T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_1T_0.75CC.csv", sep='\t')
scenario_1T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_1T_1CC.csv", sep='\t')
scenario_2T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_2T_0.5CC.csv", sep='\t')
scenario_2T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_2T_0.75CC.csv", sep='\t')
scenario_2T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_2T_1CC.csv", sep='\t')
scenario_3T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_3T_0.5CC.csv", sep='\t')
scenario_3T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_3T_0.75CC.csv", sep='\t')
scenario_3T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_3T_1CC.csv", sep='\t')
scenario_4T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_4T_0.5CC.csv", sep='\t')
scenario_4T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_4T_0.75CC.csv", sep='\t')
scenario_4T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisstate_4T_1CC.csv", sep='\t')



baseline=baseline.dropna()
#baseline_modern=modern_baseline.dropna()
scenario_1T_05CC=scenario_1T_05CC.dropna()
scenario_1T_075CC=scenario_1T_075CC.dropna()
scenario_1T_1CC=scenario_1T_1CC.dropna()
scenario_2T_05CC=scenario_2T_05CC.dropna()
scenario_2T_075CC=scenario_2T_075CC.dropna()
scenario_2T_1CC=scenario_2T_1CC.dropna()
scenario_3T_05CC=scenario_3T_05CC.dropna()
scenario_3T_075CC=scenario_3T_075CC.dropna()
scenario_3T_1CC=scenario_3T_1CC.dropna()
scenario_4T_05CC=scenario_4T_05CC.dropna()
scenario_4T_075CC=scenario_4T_075CC.dropna()
scenario_4T_1CC=scenario_4T_1CC.dropna()

baseline=baseline[baseline['x'] >= 0]

scenario_1T_05CC=scenario_1T_05CC[scenario_1T_05CC['x'] >= 0]
scenario_1T_075CC=scenario_1T_075CC[scenario_1T_075CC['x'] >= 0]
scenario_1T_1CC=scenario_1T_1CC[scenario_1T_1CC['x'] >= 0]
scenario_2T_05CC=scenario_2T_05CC[scenario_2T_05CC['x'] >= 0]
scenario_2T_075CC=scenario_2T_075CC[scenario_2T_075CC['x'] >= 0]
scenario_2T_1CC=scenario_2T_1CC[scenario_2T_1CC['x'] >= 0]
scenario_3T_05CC=scenario_3T_05CC[scenario_3T_05CC['x'] >= 0]
scenario_3T_075CC=scenario_3T_075CC[scenario_3T_075CC['x'] >= 0]
scenario_3T_1CC=scenario_3T_1CC[scenario_3T_1CC['x'] >= 0]
scenario_4T_05CC=scenario_4T_05CC[scenario_4T_05CC['x'] >= 0]
scenario_4T_075CC=scenario_4T_075CC[scenario_4T_075CC['x'] >= 0]
scenario_4T_1CC=scenario_4T_1CC[scenario_4T_1CC['x'] >= 0]






#Create a 5 year rolling average deliveries 
baseline_swpdelta=np.nanmin(baseline['x'].rolling(window=60).mean())
threshold_1T_05CC=np.nanmin(scenario_1T_05CC['x'].rolling(window=60).mean())
threshold_1T_075CC=np.nanmin(scenario_1T_075CC['x'].rolling(window=60).mean())
threshold_1T_1CC=np.nanmin(scenario_1T_1CC['x'].rolling(window=60).mean())

threshold_2T_05CC=np.nanmin(scenario_2T_05CC['x'].rolling(window=60).mean())
threshold_2T_075CC=np.nanmin(scenario_2T_075CC['x'].rolling(window=60).mean())
threshold_2T_1CC=np.nanmin(scenario_2T_1CC['x'].rolling(window=60).mean())

threshold_3T_05CC=np.nanmin(scenario_3T_05CC['x'].rolling(window=60).mean())
threshold_3T_075CC=np.nanmin(scenario_3T_075CC['x'].rolling(window=60).mean())
threshold_3T_1CC=np.nanmin(scenario_3T_1CC['x'].rolling(window=60).mean())

threshold_4T_05CC=np.nanmin(scenario_4T_05CC['x'].rolling(window=60).mean())
threshold_4T_075CC=np.nanmin(scenario_4T_075CC['x'].rolling(window=60).mean())
threshold_4T_1CC=np.nanmin(scenario_4T_1CC['x'].rolling(window=60).mean())



df = pd.DataFrame(columns=['0.5CC', '0.75CC', '1CC','min','max'], index=range(4))

df.iloc[:,0]=np.array([threshold_1T_05CC,threshold_2T_05CC,threshold_3T_05CC,threshold_4T_05CC])
df.iloc[:,1]=np.array([threshold_1T_075CC,threshold_2T_075CC,threshold_3T_075CC,threshold_4T_075CC])
df.iloc[:,2]=np.array([threshold_1T_1CC,threshold_2T_1CC,threshold_3T_1CC,threshold_4T_1CC])

df.iloc[:,3]=df.min(axis = 1)
df.iloc[:,4]=df.max(axis = 1)




datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_monthly=datDaily.resample('M').mean()

test_CDEC=datDaily_monthly['sanluisstate_S']
test_CDEC=test_CDEC.dropna()
test_CDEC=pd.DataFrame(test_CDEC)

#Create a 5 year rolling average deliveries 
test_CDEC_5_swp=np.nanmin(test_CDEC.rolling(window=60).mean())





#####################################################################################
#Read in scenarios 
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:54:55 2024

@author: rg727
"""


baseline=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_baseline.csv", sep='\t')
scenario_1T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_1T_0.5CC.csv", sep='\t')
scenario_1T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_1T_0.75CC.csv", sep='\t')
scenario_1T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_1T_1CC.csv", sep='\t')
scenario_2T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_2T_0.5CC.csv", sep='\t')
scenario_2T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_2T_0.75CC.csv", sep='\t')
scenario_2T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_2T_1CC.csv", sep='\t')
scenario_3T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_3T_0.5CC.csv", sep='\t')
scenario_3T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_3T_0.75CC.csv", sep='\t')
scenario_3T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_3T_1CC.csv", sep='\t')
scenario_4T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_4T_0.5CC.csv", sep='\t')
scenario_4T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_4T_0.75CC.csv", sep='\t')
scenario_4T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/sanluisfederal_4T_1CC.csv", sep='\t')



baseline=baseline.dropna()
scenario_1T_05CC=scenario_1T_05CC.dropna()
scenario_1T_075CC=scenario_1T_075CC.dropna()
scenario_1T_1CC=scenario_1T_1CC.dropna()
scenario_2T_05CC=scenario_2T_05CC.dropna()
scenario_2T_075CC=scenario_2T_075CC.dropna()
scenario_2T_1CC=scenario_2T_1CC.dropna()
scenario_3T_05CC=scenario_3T_05CC.dropna()
scenario_3T_075CC=scenario_3T_075CC.dropna()
scenario_3T_1CC=scenario_3T_1CC.dropna()
scenario_4T_05CC=scenario_4T_05CC.dropna()
scenario_4T_075CC=scenario_4T_075CC.dropna()
scenario_4T_1CC=scenario_4T_1CC.dropna()


baseline=baseline[baseline['x'] >= 0]

scenario_1T_05CC=scenario_1T_05CC[scenario_1T_05CC['x'] >= 0]
scenario_1T_075CC=scenario_1T_075CC[scenario_1T_075CC['x'] >= 0]
scenario_1T_1CC=scenario_1T_1CC[scenario_1T_1CC['x'] >= 0]
scenario_2T_05CC=scenario_2T_05CC[scenario_2T_05CC['x'] >= 0]
scenario_2T_075CC=scenario_2T_075CC[scenario_2T_075CC['x'] >= 0]
scenario_2T_1CC=scenario_2T_1CC[scenario_2T_1CC['x'] >= 0]
scenario_3T_05CC=scenario_3T_05CC[scenario_3T_05CC['x'] >= 0]
scenario_3T_075CC=scenario_3T_075CC[scenario_3T_075CC['x'] >= 0]
scenario_3T_1CC=scenario_3T_1CC[scenario_3T_1CC['x'] >= 0]
scenario_4T_05CC=scenario_4T_05CC[scenario_4T_05CC['x'] >= 0]
scenario_4T_075CC=scenario_4T_075CC[scenario_4T_075CC['x'] >= 0]
scenario_4T_1CC=scenario_4T_1CC[scenario_4T_1CC['x'] >= 0]





#Create a 5 year rolling average deliveries 
baseline_cvpdelta=np.nanmin(baseline['x'].rolling(window=60).mean())
threshold_1T_05CC=np.nanmin(scenario_1T_05CC['x'].rolling(window=60).mean())
threshold_1T_075CC=np.nanmin(scenario_1T_075CC['x'].rolling(window=60).mean())
threshold_1T_1CC=np.nanmin(scenario_1T_1CC['x'].rolling(window=60).mean())

threshold_2T_05CC=np.nanmin(scenario_2T_05CC['x'].rolling(window=60).mean())
threshold_2T_075CC=np.nanmin(scenario_2T_075CC['x'].rolling(window=60).mean())
threshold_2T_1CC=np.nanmin(scenario_2T_1CC['x'].rolling(window=60).mean())

threshold_3T_05CC=np.nanmin(scenario_3T_05CC['x'].rolling(window=60).mean())
threshold_3T_075CC=np.nanmin(scenario_3T_075CC['x'].rolling(window=60).mean())
threshold_3T_1CC=np.nanmin(scenario_3T_1CC['x'].rolling(window=60).mean())

threshold_4T_05CC=np.nanmin(scenario_4T_05CC['x'].rolling(window=60).mean())
threshold_4T_075CC=np.nanmin(scenario_4T_075CC['x'].rolling(window=60).mean())
threshold_4T_1CC=np.nanmin(scenario_4T_1CC['x'].rolling(window=60).mean())




df_cvpdelta = pd.DataFrame(columns=['0.5CC', '0.75CC', '1CC','min','max'], index=range(4))

df_cvpdelta.iloc[:,0]=np.array([threshold_1T_05CC,threshold_2T_05CC,threshold_3T_05CC,threshold_4T_05CC])
df_cvpdelta.iloc[:,1]=np.array([threshold_1T_075CC,threshold_2T_075CC,threshold_3T_075CC,threshold_4T_075CC])
df_cvpdelta.iloc[:,2]=np.array([threshold_1T_1CC,threshold_2T_1CC,threshold_3T_1CC,threshold_4T_1CC])
df_cvpdelta.iloc[:,3]=df_cvpdelta.min(axis = 1)
df_cvpdelta.iloc[:,4]=df_cvpdelta.max(axis = 1)



datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_monthly=datDaily.resample('M').mean()

test_CDEC=datDaily_monthly['sanluisfederal_S']
test_CDEC=test_CDEC.dropna()
test_CDEC=pd.DataFrame(test_CDEC)

#Create a 5 year rolling average deliveries 
test_CDEC_5_cvp=np.nanmin(test_CDEC.rolling(window=60).mean())






fig = plt.figure()
ax = plt.axes()
#ax.plot(range(1,5),df.iloc[:,0],color="#ee9b00")
#ax.plot(range(1,5),df.iloc[:,1],color="#ca6702")
#ax.plot(range(1,5),df.iloc[:,2],color="#ae2012")
ax.axhline(test_CDEC_5_cvp, linestyle='--', color='#0d1b2a', linewidth=2,alpha=0.6)
ax.axhline(test_CDEC_5_swp, linestyle='--', color='#778da9', linewidth=2,alpha=0.6)
ax.fill_between(range(1,5),df.iloc[:,3],df.iloc[:,4],color="#778da9",alpha=0.6)
#ax.plot(range(1,5),df_cvpdelta.iloc[:,0],color="#0a9396")
#ax.plot(range(1,5),df_cvpdelta.iloc[:,1],color="#005f73")
#ax.plot(range(1,5),df_cvpdelta.iloc[:,2],color="#001219")
ax.fill_between(range(1,5),df_cvpdelta.iloc[:,3],df_cvpdelta.iloc[:,4],color="#0d1b2a",alpha=0.6)
plt.plot(0,baseline_cvpdelta,'o',color="#0d1b2a",markersize=10) 
plt.plot(0,baseline_swpdelta,'o',color="#778da9",markersize=10) 
plt.xticks(range(0,5))
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('E:/CALFEWS-main/cc_change_heatmap/sanluisstate_sanluisfederal_new.pdf',bbox_inches='tight')



baseline=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_baseline.csv", sep='\t')
scenario_1T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_1T_0.5CC.csv", sep='\t')
scenario_1T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_1T_0.75CC.csv", sep='\t')
scenario_1T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_1T_1CC.csv", sep='\t')
scenario_2T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_2T_0.5CC.csv", sep='\t')
scenario_2T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_2T_0.75CC.csv", sep='\t')
scenario_2T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_2T_1CC.csv", sep='\t')
scenario_3T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_3T_0.5CC.csv", sep='\t')
scenario_3T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_3T_0.75CC.csv", sep='\t')
scenario_3T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_3T_1CC.csv", sep='\t')
scenario_4T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_4T_0.5CC.csv", sep='\t')
scenario_4T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_4T_0.75CC.csv", sep='\t')
scenario_4T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/oroville_4T_1CC.csv", sep='\t')



baseline=baseline.dropna()
#baseline_modern=modern_baseline.dropna()
scenario_1T_05CC=scenario_1T_05CC.dropna()
scenario_1T_075CC=scenario_1T_075CC.dropna()
scenario_1T_1CC=scenario_1T_1CC.dropna()
scenario_2T_05CC=scenario_2T_05CC.dropna()
scenario_2T_075CC=scenario_2T_075CC.dropna()
scenario_2T_1CC=scenario_2T_1CC.dropna()
scenario_3T_05CC=scenario_3T_05CC.dropna()
scenario_3T_075CC=scenario_3T_075CC.dropna()
scenario_3T_1CC=scenario_3T_1CC.dropna()
scenario_4T_05CC=scenario_4T_05CC.dropna()
scenario_4T_075CC=scenario_4T_075CC.dropna()
scenario_4T_1CC=scenario_4T_1CC.dropna()




#Create a 5 year rolling average deliveries 
baseline_swpdelta=np.nanmin(baseline['x'].rolling(window=60).sum())
threshold_1T_05CC=np.nanmin(scenario_1T_05CC['x'].rolling(window=60).sum())
threshold_1T_075CC=np.nanmin(scenario_1T_075CC['x'].rolling(window=60).sum())
threshold_1T_1CC=np.nanmin(scenario_1T_1CC['x'].rolling(window=60).sum())

threshold_2T_05CC=np.nanmin(scenario_2T_05CC['x'].rolling(window=60).sum())
threshold_2T_075CC=np.nanmin(scenario_2T_075CC['x'].rolling(window=60).sum())
threshold_2T_1CC=np.nanmin(scenario_2T_1CC['x'].rolling(window=60).sum())

threshold_3T_05CC=np.nanmin(scenario_3T_05CC['x'].rolling(window=60).sum())
threshold_3T_075CC=np.nanmin(scenario_3T_075CC['x'].rolling(window=60).sum())
threshold_3T_1CC=np.nanmin(scenario_3T_1CC['x'].rolling(window=60).sum())

threshold_4T_05CC=np.nanmin(scenario_4T_05CC['x'].rolling(window=60).sum())
threshold_4T_075CC=np.nanmin(scenario_4T_075CC['x'].rolling(window=60).sum())
threshold_4T_1CC=np.nanmin(scenario_4T_1CC['x'].rolling(window=60).sum())



df = pd.DataFrame(columns=['0.5CC', '0.75CC', '1CC','min','max'], index=range(4))

df.iloc[:,0]=np.array([threshold_1T_05CC,threshold_2T_05CC,threshold_3T_05CC,threshold_4T_05CC])
df.iloc[:,1]=np.array([threshold_1T_075CC,threshold_2T_075CC,threshold_3T_075CC,threshold_4T_075CC])
df.iloc[:,2]=np.array([threshold_1T_1CC,threshold_2T_1CC,threshold_3T_1CC,threshold_4T_1CC])

df.iloc[:,3]=df.min(axis = 1)
df.iloc[:,4]=df.max(axis = 1)




datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_monthly=datDaily.resample('M').mean()

test_CDEC=datDaily_monthly['oroville_S']
test_CDEC=test_CDEC.dropna()
test_CDEC=pd.DataFrame(test_CDEC)

#Create a 5 year rolling average deliveries 
test_CDEC_5_swp=np.nanmin(test_CDEC.rolling(window=60).mean())





#####################################################################################
#Read in scenarios 
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:54:55 2024

@author: rg727
"""


baseline=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_baseline.csv", sep='\t')
scenario_1T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_1T_0.5CC.csv", sep='\t')
scenario_1T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_1T_0.75CC.csv", sep='\t')
scenario_1T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_1T_1CC.csv", sep='\t')
scenario_2T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_2T_0.5CC.csv", sep='\t')
scenario_2T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_2T_0.75CC.csv", sep='\t')
scenario_2T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_2T_1CC.csv", sep='\t')
scenario_3T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_3T_0.5CC.csv", sep='\t')
scenario_3T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_3T_0.75CC.csv", sep='\t')
scenario_3T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_3T_1CC.csv", sep='\t')
scenario_4T_05CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_4T_0.5CC.csv", sep='\t')
scenario_4T_075CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_4T_0.75CC.csv", sep='\t')
scenario_4T_1CC=pd.read_csv("E:/CALFEWS-main/cc_change_heatmap/shasta_4T_1CC.csv", sep='\t')



baseline=baseline.dropna()
scenario_1T_05CC=scenario_1T_05CC.dropna()
scenario_1T_075CC=scenario_1T_075CC.dropna()
scenario_1T_1CC=scenario_1T_1CC.dropna()
scenario_2T_05CC=scenario_2T_05CC.dropna()
scenario_2T_075CC=scenario_2T_075CC.dropna()
scenario_2T_1CC=scenario_2T_1CC.dropna()
scenario_3T_05CC=scenario_3T_05CC.dropna()
scenario_3T_075CC=scenario_3T_075CC.dropna()
scenario_3T_1CC=scenario_3T_1CC.dropna()
scenario_4T_05CC=scenario_4T_05CC.dropna()
scenario_4T_075CC=scenario_4T_075CC.dropna()
scenario_4T_1CC=scenario_4T_1CC.dropna()




#Create a 5 year rolling average deliveries 
baseline_cvpdelta=np.nanmin(baseline['x'].rolling(window=60).sum())
threshold_1T_05CC=np.nanmin(scenario_1T_05CC['x'].rolling(window=60).sum())
threshold_1T_075CC=np.nanmin(scenario_1T_075CC['x'].rolling(window=60).sum())
threshold_1T_1CC=np.nanmin(scenario_1T_1CC['x'].rolling(window=60).sum())

threshold_2T_05CC=np.nanmin(scenario_2T_05CC['x'].rolling(window=60).sum())
threshold_2T_075CC=np.nanmin(scenario_2T_075CC['x'].rolling(window=60).sum())
threshold_2T_1CC=np.nanmin(scenario_2T_1CC['x'].rolling(window=60).sum())

threshold_3T_05CC=np.nanmin(scenario_3T_05CC['x'].rolling(window=60).sum())
threshold_3T_075CC=np.nanmin(scenario_3T_075CC['x'].rolling(window=60).sum())
threshold_3T_1CC=np.nanmin(scenario_3T_1CC['x'].rolling(window=60).sum())

threshold_4T_05CC=np.nanmin(scenario_4T_05CC['x'].rolling(window=60).sum())
threshold_4T_075CC=np.nanmin(scenario_4T_075CC['x'].rolling(window=60).sum())
threshold_4T_1CC=np.nanmin(scenario_4T_1CC['x'].rolling(window=60).sum())




df_cvpdelta = pd.DataFrame(columns=['0.5CC', '0.75CC', '1CC','min','max'], index=range(4))

df_cvpdelta.iloc[:,0]=np.array([threshold_1T_05CC,threshold_2T_05CC,threshold_3T_05CC,threshold_4T_05CC])
df_cvpdelta.iloc[:,1]=np.array([threshold_1T_075CC,threshold_2T_075CC,threshold_3T_075CC,threshold_4T_075CC])
df_cvpdelta.iloc[:,2]=np.array([threshold_1T_1CC,threshold_2T_1CC,threshold_3T_1CC,threshold_4T_1CC])
df_cvpdelta.iloc[:,3]=df_cvpdelta.min(axis = 1)
df_cvpdelta.iloc[:,4]=df_cvpdelta.max(axis = 1)



datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_monthly=datDaily.resample('M').mean()

test_CDEC=datDaily_monthly['shasta_S']
test_CDEC=test_CDEC.dropna()
test_CDEC=pd.DataFrame(test_CDEC)

#Create a 5 year rolling average deliveries 
test_CDEC_5_cvp=np.nanmin(test_CDEC.rolling(window=60).mean())






fig = plt.figure()
ax = plt.axes()
#ax.plot(range(1,5),df.iloc[:,0],color="#ee9b00")
#ax.plot(range(1,5),df.iloc[:,1],color="#ca6702")
#ax.plot(range(1,5),df.iloc[:,2],color="#ae2012")
ax.axhline(test_CDEC_5_cvp, linestyle='--', color='#0d1b2a', linewidth=2,alpha=0.6)
ax.axhline(test_CDEC_5_swp, linestyle='--', color='#778da9', linewidth=2,alpha=0.6)
ax.fill_between(range(1,5),df.iloc[:,3],df.iloc[:,4],color="#778da9",alpha=0.6)
#ax.plot(range(1,5),df_cvpdelta.iloc[:,0],color="#0a9396")
#ax.plot(range(1,5),df_cvpdelta.iloc[:,1],color="#005f73")
#ax.plot(range(1,5),df_cvpdelta.iloc[:,2],color="#001219")
ax.fill_between(range(1,5),df_cvpdelta.iloc[:,3],df_cvpdelta.iloc[:,4],color="#0d1b2a",alpha=0.6)
plt.plot(0,baseline_cvpdelta,'o',color="#0d1b2a",markersize=10) 
plt.plot(0,baseline_swpdelta,'o',color="#778da9",markersize=10) 
plt.xticks(range(0,5))
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('E:/CALFEWS-main/cc_change_heatmap/oroville_shasta_new.pdf',bbox_inches='tight')



#############################################################################################################






