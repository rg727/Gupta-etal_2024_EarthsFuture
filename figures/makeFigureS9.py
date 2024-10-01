# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:01:21 2024

@author: rg727
"""
###########################################################

#Create monthly boxplots for delta 

df_CC_spring = pd.DataFrame(columns=['March', 'April','May'], index=range(1140))
df_baseline_spring = pd.DataFrame(columns=['March', 'April','May'], index=range(2400))
modern_spring= pd.DataFrame(columns=['March', 'April','May'], index=range(20))

march=pd.read_csv("E:/CALFEWS-main/delta/delta_outflow_1550_1580_march_index.csv", sep='\t')
march=march.dropna()

df_CC = march[march['frame']=='df']
df_CC=df_CC.reset_index()


df_baseline = march[march['frame']=='df_baseline']
df_baseline=df_baseline.reset_index()

df_CC_spring['March']=df_CC['x']
df_baseline_spring['March']=df_baseline['x']


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([3])]
datDaily_monthly=datDaily.resample('M').mean()

test=pd.DataFrame(datDaily_monthly['delta_outflow'])
test=test.dropna()
test=test.reset_index()


modern_spring['March']=test['delta_outflow']

###############################################################################3
april=pd.read_csv("E:/CALFEWS-main/delta/delta_outflow_1550_1580_april_index.csv", sep='\t')
april=april.dropna()

df_CC = april[april['frame']=='df']
df_CC=df_CC.reset_index()



df_baseline = april[april['frame']=='df_baseline']
df_baseline=df_baseline.reset_index()


df_CC_spring['April']=df_CC['x']
df_baseline_spring['April']=df_baseline['x']


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([4])]
datDaily_monthly=datDaily.resample('M').mean()

test=pd.DataFrame(datDaily_monthly['delta_outflow'])
test=test.dropna()
test=test.reset_index()


modern_spring['April']=test['delta_outflow']



##################################################################################3
may=pd.read_csv("E:/CALFEWS-main/delta/delta_outflow_1550_1580_may_index.csv", sep='\t')
may=may.dropna()

df_CC = may[may['frame']=='df']
df_CC=df_CC.reset_index()



df_baseline = may[may['frame']=='df_baseline']
df_baseline=df_baseline.reset_index()



df_CC_spring['May']=df_CC['x']
df_baseline_spring['May']=df_baseline['x']


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([5])]
datDaily_monthly=datDaily.resample('M').mean()

test=pd.DataFrame(datDaily_monthly['delta_outflow'])
test=test.dropna()
test=test.reset_index()

modern_spring['May']=test['delta_outflow']

######################################################################################


data_to_plot = [test,df_baseline['x'],df_CC['x']]

def draw_plot(data, offset,edge_color, fill_color):
    pos = np.arange(data.shape[1])+offset
    bp = ax.boxplot(data, positions= pos, widths=0.1, patch_artist=True,showfliers = False)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color,alpha=0.6)

fig, ax = plt.subplots()
draw_plot(modern_spring*1233/(10**6), -0.2,"#e9d8a6","#e9d8a6")
draw_plot(df_baseline_spring.dropna()*1233/(10**6), 0, "#283618","#283618")
draw_plot(df_CC_spring*1233/(10**6), +0.2,"#bc6c25","#bc6c25")
plt.savefig('E:/CALFEWS-main/delta/delta_spring_boxplots_SI.pdf')



#plt.yscale('log')

############################################################################

df_CC_spring = pd.DataFrame(columns=['June', 'July','August'], index=range(1140))
df_baseline_spring = pd.DataFrame(columns=['June', 'July','August'], index=range(2400))
modern_spring= pd.DataFrame(columns=['June', 'July','August'], index=range(20))

june=pd.read_csv("E:/CALFEWS-main/delta/delta_outflow_1550_1580_june_index.csv", sep='\t')
june=june.dropna()

df_CC = june[june['frame']=='df']
df_CC=df_CC.reset_index()


df_baseline = june[june['frame']=='df_baseline']
df_baseline=df_baseline.reset_index()

df_CC_spring['June']=df_CC['x']
df_baseline_spring['June']=df_baseline['x']


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([6])]
datDaily_monthly=datDaily.resample('M').mean()

test=pd.DataFrame(datDaily_monthly['delta_outflow'])
test=test.dropna()
test=test.reset_index()


modern_spring['June']=test['delta_outflow']

###############################################################################3
july=pd.read_csv("E:/CALFEWS-main/delta/delta_outflow_1550_1580_july_index.csv", sep='\t')
july=july.dropna()

df_CC = july[july['frame']=='df']
df_CC=df_CC.reset_index()



df_baseline = july[july['frame']=='df_baseline']
df_baseline=df_baseline.reset_index()


df_CC_spring['July']=df_CC['x']
df_baseline_spring['July']=df_baseline['x']


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([7])]
datDaily_monthly=datDaily.resample('M').mean()

test=pd.DataFrame(datDaily_monthly['delta_outflow'])
test=test.dropna()
test=test.reset_index()


modern_spring['July']=test['delta_outflow']



##################################################################################3
august=pd.read_csv("E:/CALFEWS-main/delta/delta_outflow_1550_1580_august_index.csv", sep='\t')
august=august.dropna()

df_CC = august[august['frame']=='df']
df_CC=df_CC.reset_index()



df_baseline = august[august['frame']=='df_baseline']
df_baseline=df_baseline.reset_index()



df_CC_spring['August']=df_CC['x']
df_baseline_spring['August']=df_baseline['x']


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([8])]
datDaily_monthly=datDaily.resample('M').mean()

test=pd.DataFrame(datDaily_monthly['delta_outflow'])
test=test.dropna()
test=test.reset_index()

modern_spring['August']=test['delta_outflow']

######################################################################################


data_to_plot = [test,df_baseline['x'],df_CC['x']]

def draw_plot(data, offset,edge_color, fill_color):
    pos = np.arange(data.shape[1])+offset
    bp = ax.boxplot(data, positions= pos, widths=0.1, patch_artist=True,showfliers = False)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color,alpha=0.6)

fig, ax = plt.subplots()
draw_plot(modern_spring*1233/(10**6), -0.2,"#e9d8a6","#e9d8a6")
draw_plot(df_baseline_spring.dropna()*1233/(10**6), 0, "#283618","#283618")
draw_plot(df_CC_spring*1233/(10**6), +0.2,"#bc6c25","#bc6c25")
plt.savefig('E:/CALFEWS-main/delta/delta_summer_boxplots_SI.pdf')





######################################################################################################


df_CC_fall = pd.DataFrame(columns=['september', 'october'], index=range(1140))
df_baseline_fall = pd.DataFrame(columns=['september', 'october'], index=range(2400))
modern_fall= pd.DataFrame(columns=['september', 'october'], index=range(20))



september=pd.read_csv("E:/CALFEWS-main/delta/delta_outflow_1550_1580_september_index.csv", sep='\t')
september=september.dropna()

df_CC = september[september['frame']=='df']
df_CC=df_CC.reset_index()



df_baseline = september[september['frame']=='df_baseline']
df_baseline=df_baseline.reset_index()



df_CC_fall['september']=df_CC['x']
df_baseline_fall['september']=df_baseline['x']


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([9])]
datDaily_monthly=datDaily.resample('M').mean()

test=pd.DataFrame(datDaily_monthly['delta_outflow'])
test=test.dropna()
test=test.reset_index()

modern_fall['september']=test['delta_outflow']

######################################################################################


october=pd.read_csv("E:/CALFEWS-main/delta/delta_outflow_1550_1580_october_index.csv", sep='\t')
october=october.dropna()

df_CC = october[october['frame']=='df']
df_CC=df_CC.reset_index()



df_baseline = october[october['frame']=='df_baseline']
df_baseline=df_baseline.reset_index()



df_CC_fall['october']=df_CC['x']
df_baseline_fall['october']=df_baseline['x']


datDaily = get_results_sensitivity_number_outside_model('E:/CALFEWS-main/baseline_check/CDEC_baseline_no_changes/results.hdf5', '')
datDaily['Date']=datDaily.index
datDaily=datDaily[datDaily["Date"].dt.month.isin([10])]
datDaily_monthly=datDaily.resample('M').mean()

test=pd.DataFrame(datDaily_monthly['delta_outflow'])
test=test.dropna()
test=test.reset_index()

modern_fall['october']=test['delta_outflow']


##########################################################################################


data_to_plot = [test,df_baseline['x'],df_CC['x']]

def draw_plot(data, offset,edge_color, fill_color):
    pos = np.arange(data.shape[1])+offset
    bp = ax.boxplot(data, positions= pos, widths=0.1, patch_artist=True,showfliers = False)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color,alpha=0.6)

fig, ax = plt.subplots()
draw_plot(modern_fall*1233/(10**6), -0.2,"#e9d8a6","#e9d8a6")
draw_plot(df_baseline_fall.dropna()*1233/(10**6), 0, "#283618","#283618")
draw_plot(df_CC_fall*1233/(10**6), +0.2,"#bc6c25","#bc6c25")
plt.savefig('E:/CALFEWS-main/delta/delta_fall_boxplots_SI.pdf')