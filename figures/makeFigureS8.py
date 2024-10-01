# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:54:54 2024

@author: rg727
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:17:08 2024

@author: rg727
"""

data=pd.read_csv("E:/CALFEWS-main/ridge_plots/shasta_S_1550_1580.csv", sep='\t')
baseline=data[data['frame']=='df_baseline']
baseline=baseline.reset_index()

baseline['x']=baseline['x']*(1233/(10**6))

fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(baseline['x'], ps[j], axis=0)
    l = np.percentile(baseline['x'], ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')



ax.axhline(3438*(1233/(10**6)), linestyle='-', color='black', linewidth=2)
#ax.axhline(2774, linestyle='-', color='red', linewidth=2)
ax.axhline(1486*(1233/(10**6)), linestyle='-', color='red', linewidth=2)


ax.set_xlim(0,11)
ax.set_ylim(550*(1233/(10**6)),4552*(1233/(10**6)))
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

M = 6
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
#ax.xaxis.set_major_locator(xticks)


plt.savefig('E:/CALFEWS-main/quantiles/shasta_1550_1580_paleo_teacup_SI.pdf',bbox_inches='tight')



    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()
test_CDEC=datDaily_monthly['shasta_S']
baseline=pd.DataFrame(test_CDEC)



fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(baseline['shasta_S'], ps[j], axis=0)
    l = np.percentile(baseline['shasta_S'], ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')



ax.axhline(3438, linestyle='-', color='black', linewidth=2)
#ax.axhline(2774, linestyle='-', color='red', linewidth=2)
ax.axhline(1486, linestyle='-', color='red', linewidth=2)


ax.set_xlim(0,11)
ax.set_ylim(550,4552)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

M = 6
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
#ax.xaxis.set_major_locator(xticks)


plt.savefig('E:/CALFEWS-main/quantiles/shasta_historical_teacup.pdf',bbox_inches='tight')







####################################################################################

data=pd.read_csv("E:/CALFEWS-main/ridge_plots/oroville_deliveries_1550_1580.csv", sep='\t')
baseline=data[data['frame']=='df_baseline']
baseline=baseline.reset_index()



fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(baseline['x'], ps[j], axis=0)
    l = np.percentile(baseline['x'], ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')



ax.axhline(2250, linestyle='-', color='black', linewidth=2)
ax.axhline(852, linestyle='-', color='red', linewidth=2)



ax.set_xlim(0,11)
ax.set_ylim(852,3537)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

M = 6
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
#ax.xaxis.set_major_locator(xticks

plt.savefig('E:/CALFEWS-main/quantiles/oroville_1550_1580_paleo_teacup.pdf',bbox_inches='tight')



    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()
test_CDEC=datDaily_monthly['oroville_S']
baseline=pd.DataFrame(test_CDEC)



fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(baseline['oroville_S'], ps[j], axis=0)
    l = np.percentile(baseline['oroville_S'], ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')



ax.axhline(2250, linestyle='-', color='black', linewidth=2)
#ax.axhline(2774, linestyle='-', color='red', linewidth=2)
ax.axhline(852, linestyle='-', color='red', linewidth=2)


ax.set_xlim(0,11)
ax.set_ylim(852,3537)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

M = 6
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
#ax.xaxis.set_major_locator(xticks)


plt.savefig('E:/CALFEWS-main/quantiles/oroville_historical_teacup.pdf',bbox_inches='tight')



##########################################################################################################
data=pd.read_csv("E:/CALFEWS-main/ridge_plots/sanluisfederal_S_1550-1580.csv", sep='\t')
baseline=data[data['frame']=='df_baseline']
baseline=baseline.reset_index()



fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(baseline['x'], ps[j], axis=0)
    l = np.percentile(baseline['x'], ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')



ax.axhline(527, linestyle='-', color='black', linewidth=2)
#ax.axhline(478, linestyle='-', color='red', linewidth=2)
ax.axhline(4.8, linestyle='-', color='red', linewidth=2)



ax.set_xlim(0,11)
ax.set_ylim(26,1020)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

M = 6
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
#ax.xaxis.set_major_locator(xticks)


plt.savefig('E:/CALFEWS-main/quantiles/sanluisfederal_1550_1580_paleo_teacup.pdf', bbox_inches='tight')






    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()
test_CDEC=datDaily_monthly['sanluisfederal_S']
baseline=pd.DataFrame(test_CDEC)



fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(baseline['sanluisfederal_S'], ps[j], axis=0)
    l = np.percentile(baseline['sanluisfederal_S'], ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')




ax.axhline(527, linestyle='-', color='black', linewidth=2)
#ax.axhline(478, linestyle='-', color='red', linewidth=2)
ax.axhline(4.8, linestyle='-', color='red', linewidth=2)


ax.set_xlim(0,11)
ax.set_ylim(26,1020)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

M = 6
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
#ax.xaxis.set_major_locator(xticks)


plt.savefig('E:/CALFEWS-main/quantiles/sanluisfederal_historical_teacup.pdf',bbox_inches='tight')














############################################################################################################
data=pd.read_csv("E:/CALFEWS-main/ridge_plots/sanluisstate_S_1550-1580.csv", sep='\t')
baseline=data[data['frame']=='df']
baseline=baseline.reset_index()



fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(baseline['x'], ps[j], axis=0)
    l = np.percentile(baseline['x'], ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')



ax.axhline(606, linestyle='-', color='black', linewidth=2)
#ax.axhline(373, linestyle='-', color='red', linewidth=2)
ax.axhline(0, linestyle='-', color='red', linewidth=2)



ax.set_xlim(0,11)
ax.set_ylim(0,1020)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

M = 6
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
#ax.xaxis.set_major_locator(xticks)


plt.savefig('E:/CALFEWS-main/quantiles/sanluisstate_1550_1580_cc_teacup.pdf',bbox_inches='tight')







    
datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')

datDaily['Date']=datDaily.index

datDaily_monthly=datDaily.resample('M').mean()
test_CDEC=datDaily_monthly['sanluisstate_S']
baseline=pd.DataFrame(test_CDEC)



fig, ax = plt.subplots()

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
ps = np.arange(0, 1.01, 0.05) * 100

for j in range(1, len(ps)):
    u = np.percentile(baseline['sanluisstate_S'], ps[j], axis=0)
    l = np.percentile(baseline['sanluisstate_S'], ps[j - 1], axis=0)
    ax.fill_between(np.arange(12), l, u, color=cm.BrBG(ps[j - 1] / 100.0), alpha=0.75,
                         edgecolor='none')




ax.axhline(606, linestyle='-', color='black', linewidth=2)
#ax.axhline(478, linestyle='-', color='red', linewidth=2)
ax.axhline(0, linestyle='-', color='red', linewidth=2)


ax.set_xlim(0,11)
ax.set_ylim(26,1020)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(months)
ax.tick_params(axis='x', rotation=45)
ax.set_title('1550-1580',fontsize=14)
ax.set_ylabel('Storage (tAF)')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

M = 6
yticks = ticker.MaxNLocator(M)
xticks = ticker.MaxNLocator(M)
ax.yaxis.set_major_locator(yticks)
#ax.xaxis.set_major_locator(xticks)


plt.savefig('E:/CALFEWS-main/quantiles/sanluisstate_historical_teacup.pdf',bbox_inches='tight')




