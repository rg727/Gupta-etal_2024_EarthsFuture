# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:38:39 2024

@author: rg727
"""
#############################################################################
def convert_to_WY(row):
    if row['Date'].month>=10:
        return(pd.datetime(row['Date'].year+1,1,1).year)
    else:
        return(pd.datetime(row['Date'].year,1,1).year)
    


#############################################################################

datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([9,10])]
datDaily_monthly=datDaily.resample('M').mean()

test=pd.DataFrame(datDaily_monthly['delta_outflow'])
test=test.dropna()


threshold=np.mean(test.iloc[:,0])

#percentage_of_swp_cvp=total_releases/delta_outflow

data=pd.read_csv("E:/CALFEWS-main/delta/c.csv", sep='\t')

data=data.dropna()

df1 = data[data['frame']=='df']
np.sum(df1['x'] < threshold) /df1['x'].size  #0.228

df1 = data[data['frame']=='df_baseline']
np.sum(df1['x'] < threshold) /df1['x'].size #0.323

modern=np.sum(test<threshold)/test.size
modern


baseline=[0.7555555555555555]
CC=[0.8824561403508772]
modern=[0.65]

index = ['Fall']

color_dict = dict({'modern':'#e9d8a6','CC':'#344e41',
                  'baseline':'#3a5a40'})

#sns.set_theme(style="white")
df = pd.DataFrame({'modern':modern,'baseline': baseline,'CC': CC}, index=index)
plt.figure(figsize=(3,7))
ax = df.plot.bar(color=['#778da9','#344e41','#bc6c25'])
ax.set_ylim(0,1)
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig('E:/CALFEWS-main/delta/delta_fall_outflow.pdf')



##########################fall cummulative risk#######################


summer=pd.read_csv("E:/CALFEWS-main/delta/delta_outflow_1550_1580_fall_lineplot.csv",sep="\t")
color_dict = dict({'df_CC':'#bc6c25',
                  'df_baseline':'#344e41'})

summer.columns=['Index1','Index2','Probability','Scenario']
fig,ax = plt.subplots()
sns.lineplot(data=summer, x="Index2", y="Probability",hue='Scenario',palette=color_dict,err_kws={'alpha':0.6})
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
from matplotlib import ticker
M = 6
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
ax.xaxis.set_major_locator(xticks)
plt.savefig('E:/CALFEWS-main/delta/delta_outflow_fall_lineplot.pdf',bbox_inches='tight')


########################################################################

baseline=[0.99,0.78]
CC=[1,0.85]
modern=[0.75,0.75]

index = ['74 km','81 km']

color_dict = dict({'modern':'#e9d8a6','CC':'#344e41',
                  'baseline':'#3a5a40'})

#sns.set_theme(style="white")
df = pd.DataFrame({'modern':modern,'baseline': baseline,'CC': CC}, index=index)
plt.figure(figsize=(3,7))
ax = df.plot.bar(color=['#778da9','#344e41','#bc6c25'])
ax.set_ylim(0,1)
plt.legend([],[], frameon=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('E:/CALFEWS-main/delta/delta_salinity.pdf',bbox_inches='tight')

##########################################################################
