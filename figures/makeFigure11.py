# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:31:22 2024

@author: rg727
"""


#Tehachapi 

    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily['WaterYear'] = datDaily.apply(lambda x: convert_to_WY(x), axis=1)


datDaily_annual_deliveries_baseline=datDaily.groupby(['WaterYear'])['tehachapi_tableA_delivery'].max()

datDaily_annual_demand_baseline=datDaily.groupby(['WaterYear'])['tehachapi_tot_demand'].max()

shortage_baseline=(datDaily_annual_demand_baseline-datDaily_annual_deliveries_baseline)/datDaily_annual_demand_baseline


shortage_baseline_truncated=shortage_baseline[15:20]


data=pd.read_csv("E:/CALFEWS-main/individual_users/tehachapi_1550_1580_WY.csv", sep='\t')

data['x']=1-data['x']



color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig, ax = plt.subplots()

                  
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

#plt.axvline(x=0.99, color="red",ls=':',ymin=0, ymax=0.95)
plt.axvline(0.01, color='red',ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=(1-np.mean(shortage_baseline)), color='black',ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0,1.0) 

ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.tick_params(axis='x', pad=15)

plt.savefig('E:/CALFEWS-main/individual_users/tehachapi_1550_1580_min.pdf')

#Northern Kern 

    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily['WaterYear'] = datDaily.apply(lambda x: convert_to_WY(x), axis=1)


datDaily_annual_deliveries_baseline=datDaily.groupby(['WaterYear'])['northkern_kern_delivery'].max()

datDaily_annual_demand_baseline=datDaily.groupby(['WaterYear'])['northkern_tot_demand'].max()

shortage_baseline=(datDaily_annual_demand_baseline-datDaily_annual_deliveries_baseline)/datDaily_annual_demand_baseline


shortage_baseline_truncated=shortage_baseline[15:20]


data=pd.read_csv("E:/CALFEWS-main/ridge_plots/northkern_1550_1580_april.csv", sep='\t')



data['x']=1-data['x']


color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})


fig, ax = plt.subplots()
                  
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

plt.axvline(x=(1-np.max(shortage_baseline_truncated)), color='red',ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=(1-np.mean(shortage_baseline)), color='black',ymin=0, ymax=0.05,linewidth=4)

plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0,1.0) 


plt.savefig('E:/CALFEWS-main/individual_users/northernkern_1550_1580_min.pdf')


###############################################################################################
#KRWA

    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily['WaterYear'] = datDaily.apply(lambda x: convert_to_WY(x), axis=1)


datDaily_annual_deliveries_baseline=datDaily.groupby(['WaterYear'])['krwa_kings_delivery'].max()

datDaily_annual_demand_baseline=datDaily.groupby(['WaterYear'])['krwa_tot_demand'].max()

shortage_baseline=(datDaily_annual_demand_baseline-datDaily_annual_deliveries_baseline)/datDaily_annual_demand_baseline


shortage_baseline_truncated=shortage_baseline[15:20]


data=pd.read_csv("E:/CALFEWS-main/ridge_plots/krwa_1550_1580_april.csv", sep='\t')


data['x']=1-data['x']


color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})

from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker


fig, ax = plt.subplots()



                  
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

#plt.axvline(x=np.mean(shortage_baseline_truncated), color='red',ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=(1-np.mean(shortage_baseline)), color='black',ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=(1-np.max(shortage_baseline_truncated)), color='red',ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0,0.3) 
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.tick_params(axis='x', pad=15)
plt.savefig('E:/CALFEWS-main/individual_users/krwa_1550_1580_min.pdf')


###############################################################################################


#Tule

    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily['WaterYear'] = datDaily.apply(lambda x: convert_to_WY(x), axis=1)


datDaily_annual_deliveries_baseline=datDaily.groupby(['WaterYear'])['othertule_tule_delivery'].max()

datDaily_annual_demand_baseline=datDaily.groupby(['WaterYear'])['othertule_tot_demand'].max()

shortage_baseline=(datDaily_annual_demand_baseline-datDaily_annual_deliveries_baseline)/datDaily_annual_demand_baseline


shortage_baseline_truncated=shortage_baseline[15:20]


data=pd.read_csv("E:/CALFEWS-main/individual_users/tule_1550_1580_WY.csv", sep='\t')
data['x']=1-data['x']


fig, ax = plt.subplots()


color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
                  
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

#plt.axvline(x=0.99, color="red",ls=':',ymin=0, ymax=0.95)
#plt.axvline(x=np.mean(shortage_baseline_truncated), color='red',ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=(1-np.max(shortage_baseline_truncated)), color='red',ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=(1-np.mean(shortage_baseline)), color='black',ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.tick_params(axis='x', pad=15)
plt.savefig('E:/CALFEWS-main/individual_users/tule_1550_1580_min.pdf')


#################################################################################################

#Friant

    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index
datDaily['WaterYear'] = datDaily.apply(lambda x: convert_to_WY(x), axis=1)


datDaily_annual_deliveries_baseline=datDaily.groupby(['WaterYear'])['delano_friant1_delivery'].max()

datDaily_annual_demand_baseline=datDaily.groupby(['WaterYear'])['delano_tot_demand'].max()

shortage_baseline=(datDaily_annual_demand_baseline-datDaily_annual_deliveries_baseline)/datDaily_annual_demand_baseline


shortage_baseline_truncated=shortage_baseline[15:20]


data=pd.read_csv("E:/CALFEWS-main/individual_users/delano_friant1_1550_1580_WY.csv", sep='\t')

data['x']=1-data['x']



color_dict = dict({'df':'#BC6C25',
                  'df_baseline':'#283618'})
fig, ax = plt.subplots()                 
sns.kdeplot(
   data=data, x="x", hue="frame",
   fill=True, common_norm=False, palette=color_dict,
   alpha=.5, linewidth=0
)                  

#plt.axvline(x=0.99, color="red",ls=':',ymin=0, ymax=0.95)
#plt.axvline(x=np.mean(shortage_baseline_truncated), color='red',ymin=0, ymax=0.05,linewidth=4)
plt.axvline(1-0.8, color='red',ymin=0, ymax=0.05,linewidth=4)
plt.axvline(x=(1-np.mean(shortage_baseline)), color='black',ymin=0, ymax=0.05,linewidth=4)
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
plt.xlim(0,1) 
ax.tick_params(axis='x', pad=15)

plt.savefig('E:/CALFEWS-main/individual_users/delano_friant1_1550_1580_min.pdf')


