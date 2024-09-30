# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:13:29 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:11:50 2024

@author: rg727

"""


from matplotlib import ticker 
from matplotlib.ticker import FormatStrFormatter
#CVP: Shasta

    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()
test_CDEC=datDaily_monthly['shasta_S']






data=pd.read_csv("E:/CALFEWS-main/ridge_plots/shasta_S_1550_1580.csv", sep='\t')


data['x']=data['x']*(1233/(10**6))



color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig,ax = plt.subplots()              
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  


##plt.axvline(x=1486, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(x=np.mean(test_CDEC*1233/(10**6)), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=np.min(test_CDEC*1233/(10**6)), color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(550*(1233/(10**6)),4552*(1233/(10**6))) 
plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/SHA_1550_1580_min_SI.pdf',bbox_inches='tight')






color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig,ax = plt.subplots()              
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)    

#plt.axvline(x=1486, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(x=np.mean(test_CDEC), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=2774, color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/SHA_1550_1580.pdf',bbox_inches='tight')


###############################################################################################



#CVP: San Luis Federal
    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()
test_CDEC=datDaily_monthly['sanluisfederal_S']





datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date'] =datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([4])]
datDaily_monthly=datDaily.resample('M').mean()

datDaily_monthly_truncated=datDaily_monthly['2011-10-31': '2016-09-30']

#Isolate daily data
#test=datDaily_monthly_truncated['delta_outflow']
test=datDaily_monthly_truncated['sanluisfederal_S']
np.mean(test)



data=pd.read_csv("E:/CALFEWS-main/ridge_plots/sanluisfederal_S_1550-1580.csv", sep='\t')


data['x']=data['x']*(1233/(10**6))


color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})




color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig,ax = plt.subplots()              
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

'''
sns.kdeplot(
   data=pd.DataFrame(test_CDEC), x="sanluisfederal_S",
   fill=False, common_norm=False, color='black', linewidth=2,linestyle="--"
)
'''
ax.tick_params(axis='x', pad=15)
#plt.axvline(x=1486, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(x=np.mean(test_CDEC*(1233/(10**6))), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=np.min(test_CDEC*(1233/(10**6))), color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(0,1020*(1233/(10**6))) 

plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/sanluisfederal_1550_1580_min_SI.pdf',bbox_inches='tight')




                  
fig,ax = plt.subplots()                         
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

#plt.axvline(x=1486, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(np.mean(test_CDEC), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(np.mean(test), color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
ax.yaxis.set_major_locator(yticks)
M = 
yticks = ticker.MaxNLocator(M)
ax.xaxis.set_major_locator(yticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/sanluisfederal_1550_1580.pdf',bbox_inches='tight')




#CVP: San Luis State

datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()
test_CDEC=datDaily_monthly['sanluisstate_S']




data=pd.read_csv("E:/CALFEWS-main/ridge_plots/sanluisstate_S_1550-1580.csv", sep='\t')

datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date'] =datDaily.index
#datDaily=datDaily[datDaily["Date"].dt.month.isin([4])]
datDaily_monthly=datDaily.resample('M').mean()

datDaily_monthly_truncated=datDaily_monthly['2011-10-31': '2016-09-30']

#Isolate daily data
#test=datDaily_monthly_truncated['delta_outflow']
test=datDaily_monthly_truncated['sanluisstate_S']
np.mean(test)



data['x']=data['x']*(1233/(10**6))



color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig,ax = plt.subplots()              
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

'''
sns.kdeplot(
   data=pd.DataFrame(test_CDEC), x="sanluisstate_S",
   fill=False, common_norm=False, color='black', linewidth=2,linestyle="--"
)

'''

#plt.axvline(x=1486, color="red",ls=':',ymin=0, ymax=0.95)
ax.tick_params(axis='x', pad=15)
plt.axvline(x=np.mean(test_CDEC*(1233/(10**6))), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(10**(1233/(10**6)), color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(0,1020*(1233/(10**6))) 
plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/sanluisstate_1550_1580_min_SI.pdf',bbox_inches='tight')






#CVP Exchange

datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_truncated=datDaily[datDaily["Date"].dt.month.isin([9])]

datDaily_monthly=datDaily_truncated.resample('M').mean()

test_CDEC=datDaily_monthly['exchange_contract']





datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date'] =datDaily.index
columns=list(datDaily.columns)
#datDaily=datDaily[datDaily["Date"].dt.month.isin([4])]
datDaily_monthly=datDaily.resample('M').mean()

datDaily_monthly_truncated=datDaily_monthly['2011-10-31': '2016-09-30']
datDaily_monthly_truncated=datDaily_monthly_truncated[datDaily["Date"].dt.month.isin([9])]

#Isolate daily data
#test=datDaily_monthly_truncated['delta_outflow']
test=datDaily_monthly_truncated['exchange_contract']
np.mean(test)



data=pd.read_csv("E:/CALFEWS-main/ridge_plots/CVP_exchange_summer_1550_1580.csv", sep='\t')

data['x']=data['x']*(1233/(10**6))

color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig,ax = plt.subplots()              
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  



#plt.axvline(x=1486, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(x=np.mean(test_CDEC*(1233/(10**6))), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=np.min(test_CDEC*(1233/(10**6))), color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/cvp_exhange_min_SI.pdf',bbox_inches='tight')




fig = plt.figure()
ax = plt.axes()

color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
                  
                  
                  
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  
plt.legend([],[], frameon=False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.axvline(np.mean(test_CDEC), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=1042, color="red",ymin=0, ymax=0.05,linewidth=4)
M = 5
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/cvp_exhange.pdf',bbox_inches='tight')

df1 = data[data['frame']=='df_baseline']
df1=df1.dropna()
np.sum(df1['x'] < np.mean(test)) /df1['x'].size # 0.99 

df1 = data[data['frame']=='df']
df1=df1.dropna()
np.sum(df1['x'] < np.mean(test)) /df1['x'].size # 0.99 


df1 = data[data['frame']=='df_baseline']
df1=df1.dropna()
np.sum(df1['x'] <900) /df1['x'].size # 0.99 

df1 = data[data['frame']=='df']
df1=df1.dropna()
np.sum(df1['x'] < 900) /df1['x'].size # 0.99 




#CVP Delta


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_truncated=datDaily[datDaily["Date"].dt.month.isin([9])]

datDaily_monthly=datDaily_truncated.resample('M').mean()

test_CDEC=datDaily_monthly['cvpdelta_contract']





datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date'] =datDaily.index
columns=list(datDaily.columns)
#datDaily=datDaily[datDaily["Date"].dt.month.isin([4])]
datDaily_monthly=datDaily.resample('M').mean()

datDaily_monthly_truncated=datDaily_monthly['2011-10-31': '2016-09-30']
datDaily_monthly_truncated=datDaily_monthly_truncated[datDaily["Date"].dt.month.isin([9])]

#Isolate daily data
#test=datDaily_monthly_truncated['delta_outflow']
test=datDaily_monthly_truncated['cvpdelta_contract']
np.mean(test)



data=pd.read_csv("E:/CALFEWS-main/ridge_plots/CVP_delta_summer_1550_1580.csv", sep='\t')
data['x']=data['x']*(1233/(10**6))

color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig,ax = plt.subplots()              
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

'''
sns.kdeplot(
   data=pd.DataFrame(test_CDEC), x="cvpdelta_contract",
   fill=False, common_norm=False, color='black', linewidth=2,linestyle="--"
)

'''

#plt.axvline(x=1486, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(x=np.mean(test_CDEC*(1233/(10**6))), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=np.min(test_CDEC*(1233/(10**6))), color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/cvp_delta_min_SI.pdf',bbox_inches='tight')



#SWP Delta

datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily_truncated=datDaily[datDaily["Date"].dt.month.isin([9])]

datDaily_monthly=datDaily_truncated.resample('M').mean()

test_CDEC=datDaily_monthly['tableA_contract']




datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date'] =datDaily.index
columns=list(datDaily.columns)
#datDaily=datDaily[datDaily["Date"].dt.month.isin([4])]
datDaily_monthly=datDaily.resample('M').mean()

datDaily_monthly_truncated=datDaily_monthly['2011-10-31': '2016-09-30']
datDaily_monthly_truncated=datDaily_monthly_truncated[datDaily["Date"].dt.month.isin([9])]

#Isolate daily data
#test=datDaily_monthly_truncated['delta_outflow']
test=datDaily_monthly_truncated['tableA_contract']
np.mean(test)



data=pd.read_csv("E:/CALFEWS-main/ridge_plots/swpdelta_contract_1550_1580.csv", sep='\t')
data['x']=data['x']*(1233/(10**6))


color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig,ax = plt.subplots()              
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  


ax.tick_params(axis='x', pad=15)
plt.xlim(left=0) 
#plt.axvline(x=1486, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(x=np.mean(test_CDEC*(1233/(10**6))), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=np.min(test_CDEC*(1233/(10**6))), color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/swp_delta_min_SI.pdf',bbox_inches='tight')





#Oroville

datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()

test_CDEC=datDaily_monthly['oroville_S']



datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date'] =datDaily.index
datDaily_monthly=datDaily.resample('M').mean()
datDaily_monthly_truncated=datDaily_monthly['2011-10-31': '2016-09-30']

#Isolate daily data
#test=datDaily_monthly_truncated['delta_outflow']
test=datDaily_monthly_truncated['oroville_S']
np.mean(test)


data=pd.read_csv("E:/CALFEWS-main/ridge_plots/oroville_deliveries_1550_1580.csv", sep='\t')

data['x']=data['x']*(1233/(10**6))


color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig,ax = plt.subplots()              
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

'''
sns.kdeplot(
   data=pd.DataFrame(test_CDEC), x="oroville_S",
   fill=False, common_norm=False, color='black', linewidth=2,linestyle="--"
)
'''

#plt.axvline(x=1486, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(x=np.mean(test_CDEC*(1233/(10**6))), color="black",ymin=0, ymax=0.05,linewidth=4)
plt.axvline(870*(1233/(10**6)), color="red",ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
M = 4
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(850*(1233/(10**6)),3537*(1233/(10**6))) 

plt.savefig('E:/CALFEWS-main/cvp_swp_cascade/ORO_1550_1580_min.pdf',bbox_inches='tight')


