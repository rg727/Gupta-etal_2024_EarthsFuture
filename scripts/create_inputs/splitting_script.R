#Splitting script to create datasets 

#Paleo + snow setup 
for (j in 7:56){

setwd("./Input_Data/hydroclimate/Paleo/")


data=read.table("ORO_snow.txt",header=TRUE)



#Date from 10/1/1400 to 9/30/2016

#Create a date from three columns

created_date=format(seq(as.Date("1399-11-01"), as.Date("2017-04-30"), by="days"), format="%m-%d-%Y")



datetime <- as.Date(with(data, paste(data$sim_datemat_2, data$sim_datemat_3,data$sim_datemat_1,sep="-")), "%m-%d-%Y")

data$datetime=gsub(" 0", " ",format(datetime,"%m/%d/%Y"))

gsub(" 0", " ", format(as.Date("1998-09-02"), "%Y, %m, %d"))

#Rearrange data 

data$realization=1



all_datasets=data.frame(data$datetime, data$realization,data[,j])

data=read.table("SHA_snow.txt",header=TRUE)
all_datasets$SHA_snow=data[,j]

data=read.table("FOL_snow.txt",header=TRUE)
all_datasets$FOL_snow=data[,j]

data=read.table("YRS_snow.txt",header=TRUE)
all_datasets$YRS_snow=data[,j]

data=read.table("NML_snow.txt",header=TRUE)
all_datasets$NML_snow=data[,j]

data=read.table("TLG_snow.txt",header=TRUE)
all_datasets$DNP_snow=data[,j]

data=read.table("EXC_snow.txt",header=TRUE)
all_datasets$EXC_snow=data[,j]

data=read.table("MIL_snow.txt",header=TRUE)
all_datasets$MIL_snow=data[,j]

data=read.table("PNF_snow.txt",header=TRUE)
all_datasets$PNF_snow=data[,j]

data=read.table("TRM_snow.txt",header=TRUE)
all_datasets$KWH_snow=data[,j]

data=read.table("SUC_snow.txt",header=TRUE)
all_datasets$SUC_snow=data[,j]

data=read.table("ISB_snow.txt",header=TRUE)
all_datasets$ISB_snow=data[,j]

all_datasets[,3:14]=all_datasets[,3:14]*0.0393701 #change to inches 

data=read.table("ORO_baseline.txt",header=TRUE)
all_datasets$ORO_fnf=data[,j]

data=read.table("SHA_baseline.txt",header=TRUE)
all_datasets$SHA_fnf=data[,j]

data=read.table("FOL_baseline.txt",header=TRUE)
all_datasets$FOL_fnf=data[,j]

data=read.table("YRS_baseline.txt",header=TRUE)
all_datasets$YRS_fnf=data[,j]

data=read.table("NML_baseline.txt",header=TRUE)
all_datasets$NML_fnf=data[,j]


data=read.table("TLG_baseline.txt",header=TRUE)
all_datasets$DNP_fnf=data[,j]


data=read.table("EXC_baseline.txt",header=TRUE)
all_datasets$EXC_fnf=data[,j]

data=read.table("MIL_baseline.txt",header=TRUE)
all_datasets$MIL_fnf=data[,j]

data=read.table("PNF_baseline.txt",header=TRUE)
all_datasets$PFT_fnf=data[,j]

data=read.table("TRM_baseline.txt",header=TRUE)
all_datasets$KWH_fnf=data[,j]

data=read.table("SUC_baseline.txt",header=TRUE)
all_datasets$SUC_fnf=data[,j]

data=read.table("ISB_baseline.txt",header=TRUE)
all_datasets$ISB_fnf=data[,j]


#Rename the columns 
colnames(all_datasets)=c('datetime','realization','ORO_snow','SHA_snow','FOL_snow','YRS_snow','NML_snow','DNP_snow','EXC_snow','MIL_snow','PFT_snow','KWH_snow','SUC_snow','ISB_snow','ORO_fnf','SHA_fnf','FOL_fnf','YRS_fnf','NML_fnf','DNP_fnf','EXC_fnf','MIL_fnf','PFT_fnf','KWH_fnf','SUC_fnf','ISB_fnf')



#Right now fnf is in mm/day, but we need to convert to acre-feet per day 

area_ft2_FOL = 1885*5280*5280
area_ft2_MIL = 1675*5280*5280
area_ft2_ISB = 2074*5280*5280
area_ft2_MRC = 1061*5280*5280
area_ft2_NML = 900*5280*5280
area_ft2_ORO = 3607*5280*5280
area_ft2_PNF = 1545*5280*5280
area_ft2_SHA = 6665*5280*5280
area_ft2_SCC = 393*5280*5280
area_ft2_TRM = 561*5280*5280
area_ft2_TLG = 1538*5280*5280
area_ft2_YRS = 1108*5280*5280


all_datasets$ORO_fnf=all_datasets$ORO_fnf*area_ft2_ORO/304.8/43560
all_datasets$FOL_fnf=all_datasets$FOL_fnf*area_ft2_FOL/304.8/43560
all_datasets$MIL_fnf=all_datasets$MIL_fnf*area_ft2_MIL/304.8/43560
all_datasets$ISB_fnf=all_datasets$ISB_fnf*area_ft2_ISB/304.8/43560
all_datasets$EXC_fnf=all_datasets$EXC_fnf*area_ft2_MRC/304.8/43560
all_datasets$NML_fnf=all_datasets$NML_fnf*area_ft2_NML/304.8/43560
all_datasets$PFT_fnf=all_datasets$PFT_fnf*area_ft2_PNF/304.8/43560
all_datasets$SHA_fnf=all_datasets$SHA_fnf*area_ft2_SHA/304.8/43560
all_datasets$SUC_fnf=all_datasets$SUC_fnf*area_ft2_SCC/304.8/43560
all_datasets$KWH_fnf=all_datasets$KWH_fnf*area_ft2_TRM/304.8/43560
all_datasets$DNP_fnf=all_datasets$DNP_fnf*area_ft2_TLG/304.8/43560
all_datasets$YRS_fnf=all_datasets$YRS_fnf*area_ft2_YRS/304.8/43560


###############################################################Splitting script###################################################

ensemble_member = j-6


#batch_1=all_datasets[335:11291,3:ncol(all_datasets)]
#write.csv(batch_1,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/1/1.csv"),row.names = FALSE)


#batch_2=all_datasets[11292:22249,3:ncol(all_datasets)]
#write.csv(batch_2,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/2/2.csv"),row.names = FALSE)


#batch_3=all_datasets[22250:33206,3:ncol(all_datasets)]
#write.csv(batch_3,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/3/3.csv"),row.names = FALSE)

#batch_4=all_datasets[33207:44163,3:ncol(all_datasets)]
#write.csv(batch_4,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/4/4.csv"),row.names = FALSE)


#batch_5=all_datasets[44164:55120,3:ncol(all_datasets)]
#write.csv(batch_5,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/5/5.csv"),row.names = FALSE)

batch_6=all_datasets[55121:66078,3:ncol(all_datasets)]
write.csv(batch_6,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/6/6.csv"),row.names = FALSE)

#batch_7=all_datasets[66079:77035,3:ncol(all_datasets)]
#write.csv(batch_7,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/7/7.csv"),row.names = FALSE)

#batch_8=all_datasets[77036:87993,3:ncol(all_datasets)]
#write.csv(batch_8,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/8/8.csv"),row.names = FALSE)

#batch_9=all_datasets[87994:98950,3:ncol(all_datasets)]
#write.csv(batch_9,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/9/9.csv"),row.names = FALSE)


# batch_10=all_datasets[98951:109907,3:ncol(all_datasets)]
# write.csv(batch_10,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/10/10.csv"),row.names = FALSE)

# batch_11=all_datasets[109908:120864,3:ncol(all_datasets)]
# write.csv(batch_11,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/11/11.csv"),row.names = FALSE)

# batch_12=all_datasets[120865:131822,3:ncol(all_datasets)]
# write.csv(batch_12,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/12/12.csv"),row.names = FALSE)

# batch_13=all_datasets[131823:142779,3:ncol(all_datasets)]
# write.csv(batch_13,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/13/13.csv"),row.names = FALSE)

# batch_14=all_datasets[142780:153736,3:ncol(all_datasets)]
# write.csv(batch_14,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/14/14.csv"),row.names = FALSE)

# batch_15=all_datasets[153737:164693,3:ncol(all_datasets)]
# write.csv(batch_15,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/15/15.csv"),row.names = FALSE)

# batch_16=all_datasets[164694:175651,3:ncol(all_datasets)]
# write.csv(batch_16,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/16/16.csv"),row.names = FALSE)

# batch_17=all_datasets[175652:186607,3:ncol(all_datasets)]
# write.csv(batch_17,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/17/17.csv"),row.names = FALSE)

#batch_18=all_datasets[186608:197565,3:ncol(all_datasets)]
#write.csv(batch_18,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/18/18.csv"),row.names = FALSE)

# batch_19=all_datasets[197566:208522,3:ncol(all_datasets)]
# write.csv(batch_19,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/19/19.csv"),row.names = FALSE)

#batch_20=all_datasets[208523:219480,3:ncol(all_datasets)]
#write.csv(batch_20,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/20/20.csv"),row.names = FALSE)

#batch_21=all_datasets[219481:225536,3:ncol(all_datasets)]
#write.csv(batch_21,paste0("/home/fs02/pmr82_0001/rg727/CALFEWS/",ensemble_member,"/21/21.csv"),row.names = FALSE)
}
