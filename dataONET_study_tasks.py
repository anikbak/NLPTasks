########################################################################################################
# STUDY TASKS IN O*NET DATASET
########################################################################################################

import numpy as np, pandas as pd 
level = 4 # 0

# Load Data
if level == 0:
    cols = ['O*NET-SOC Code', 'Task', 'VerbNouns_dobj',
       'Occupation_VerbNouns', 'OrigTaskIndex', 'Task ID', 'Task Type',
       'Incumbents Responding', 'Domain Source']
    df = pd.read_csv("data/data_onet_task_verbnoun_pairs.csv",usecols=cols)
elif level == 4:
    cols = ['O*NET-SOC Code', 'Task', 'VerbNouns_dobj_NounLevel_4',
       'Occupation_VerbNouns', 'OrigTaskIndex', 'Task ID', 'Task Type',
       'Incumbents Responding', 'Domain Source']
    df = pd.read_csv("data/data_onet_task_nounlevel4_verbnoun_pairs.csv")

# Construct Importance Measures for Occupational Tasks
colsImp = ['O*NET-SOC Code', 'Task ID', 'Scale ID', 'Category', 'Data Value', 'N']
dfImp = pd.read_csv("data_raw/onet/Task Ratings.txt",sep='\t',usecols=colsImp)

dfImpIM = dfImp.loc[dfImp['Scale ID']=='IM'].copy()
dfImpIM['Data Value'] = (dfImpIM['Data Value']-1) * 100 / 4
dfImpIM = dfImpIM.drop(columns=['Category','Scale ID','N']).rename(columns={'Data Value':'Data Value IM'})

dfImpRT = dfImp.loc[dfImp['Scale ID']=='RT'].copy()
dfImpRT['Data Value'] = (dfImpRT['Data Value']-10) * 100 / 90
dfImpRT = dfImpRT.drop(columns=['Category','Scale ID','N']).rename(columns={'Data Value':'Data Value RT'})

dfImpFT_ = dfImp.loc[dfImp['Scale ID']=='FT'].copy() 
dfImpFT = dfImpFT_.loc[dfImpFT_['Category']==1].copy()
dfImpFT = dfImpFT.rename(columns={'Data Value': 'Data Value, FT Category 1'}).drop(columns=['Category','Scale ID','N'])

for i in range(2,8):
    temp = dfImpFT_.loc[dfImpFT_['Category']==i].copy()
    temp = temp.rename(columns={'Data Value': 'Data Value, FT Category '+str(i)}).drop(columns=['Category','Scale ID','N'])
    dfImpFT = dfImpFT.merge(temp,on=['O*NET-SOC Code', 'Task ID'],how='outer')

dfImp_ = dfImpRT.copy() 
dfImp_ = dfImp_.merge(dfImpIM,on=['O*NET-SOC Code', 'Task ID'],how='outer')
dfImp_ = dfImp_.merge(dfImpFT,on=['O*NET-SOC Code', 'Task ID'],how='outer')

# Construct Importance Indices
dfImp_ = dfImp_.set_index(['O*NET-SOC Code','Task ID'])
dfImp_[['Weight RT','Weight IM']] = dfImp_.groupby(['O*NET-SOC Code','Task ID'])[['Data Value RT','Data Value IM']].mean()
dfImp_ = dfImp_.reset_index() 

# Add Occupation Titles and Importance Measures to Occupational Tasks
dfNam = pd.read_csv("data_raw/onet/Occupation Data.txt",sep='\t')

df = df.merge(dfImp_,on=['O*NET-SOC Code','Task ID'],how='outer',indicator='_m_occRatings')
df = df.merge(dfNam,on=['O*NET-SOC Code'],how='outer')
df = df.drop(columns='Unnamed: 0')

# Export 
df.to_csv('data/data_onet_weights.csv')
