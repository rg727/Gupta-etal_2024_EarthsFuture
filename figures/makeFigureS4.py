# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

years=np.tile(range(1400,2017), 100)

#Import WR fractions

WR_fractions=pd.read_csv("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR_fractions.csv")


#Create line plot 

d={'Year':years, 'Frequency':WR_fractions.iloc[:,0]}


WR_1=pd.DataFrame(data=d)
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 16, "ytick.major.size": 16})
lineplot=sns.lineplot(data=WR_1, x="Year", y="Frequency",color='#bb9457')
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR1_whole.pdf",bbox_inches='tight')
lineplot=sns.lineplot(data=WR_1, x="Year", y="Frequency",color='#bb9457')
lineplot.set_xlim(1550,1580)
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR1_truncated.pdf",bbox_inches='tight')


d={'Year':years, 'Frequency':WR_fractions.iloc[:,1]}


WR_2=pd.DataFrame(data=d)
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 16, "ytick.major.size": 16})
lineplot=sns.lineplot(data=WR_2, x="Year", y="Frequency",color='#606c38')
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR2_whole.pdf",bbox_inches='tight')
lineplot=sns.lineplot(data=WR_2, x="Year", y="Frequency",color='#606c38')
lineplot.set_xlim(1550,1580)
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR2_truncated.pdf",bbox_inches='tight')



d={'Year':years, 'Frequency':WR_fractions.iloc[:,2]}


WR_3=pd.DataFrame(data=d)
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 16, "ytick.major.size": 16})
lineplot=sns.lineplot(data=WR_3, x="Year", y="Frequency",color='#bc6c25')
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR3_whole.pdf",bbox_inches='tight')
lineplot=sns.lineplot(data=WR_3, x="Year", y="Frequency",color='#bc6c25')
lineplot.set_xlim(1550,1580)
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR3_truncated.pdf",bbox_inches='tight')



d={'Year':years, 'Frequency':WR_fractions.iloc[:,3]}


WR_4=pd.DataFrame(data=d)
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 16, "ytick.major.size": 16})
lineplot=sns.lineplot(data=WR_4, x="Year", y="Frequency",color='#dda15e')
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR4_whole.pdf",bbox_inches='tight')
lineplot=sns.lineplot(data=WR_4, x="Year", y="Frequency",color='#dda15e')
lineplot.set_xlim(1550,1580)
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR4_truncated.pdf",bbox_inches='tight')



d={'Year':years, 'Frequency':WR_fractions.iloc[:,4]}


WR_5=pd.DataFrame(data=d)
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 16, "ytick.major.size": 16})
lineplot=sns.lineplot(data=WR_5, x="Year", y="Frequency",color='#283618')
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR5_whole.pdf",bbox_inches='tight')
lineplot=sns.lineplot(data=WR_5, x="Year", y="Frequency",color='#283618')
lineplot.set_xlim(1550,1580)
plt.savefig("C:/Users/rg727/Documents/PhD/Weather_Regimes/WR5_truncated.pdf",bbox_inches='tight')


