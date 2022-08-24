########################################################################################################
# STUDY TASKS IN O*NET DATASET
########################################################################################################

import numpy as np, pandas as pd, itertools, time
from joblib import Parallel, delayed 

# Choose Level of Generality for Task Nouns. 
# If level == 0, uses the raw Task nouns. 
# If level == 4, uses nouns at generality level 4. 
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

# Add Occupation Titles and Importance Measures to Occupational Tasks
dfNam = pd.read_csv("data_raw/onet/Occupation Data.txt",sep='\t')

df = df.merge(dfImp_,on=['O*NET-SOC Code','Task ID'],how='outer',indicator='_m_occRatings')
df = df.merge(dfNam,on=['O*NET-SOC Code'],how='outer')
df = df.drop(columns='Unnamed: 0')
df = df.loc[pd.notna(df['Task'])].copy()

# Export 
df.to_csv('data/data_onet_weights.csv')

# Construct Dataset at occupation x task level
df = df.set_index(['O*NET-SOC Code'])
df['count_total'] = df.groupby(['O*NET-SOC Code'])['Occupation_VerbNouns'].count()  
df = df.reset_index()

df = df.set_index(['O*NET-SOC Code','Occupation_VerbNouns'])
df['count'] = df.groupby(['O*NET-SOC Code','Occupation_VerbNouns'])['Task'].count()
df[['WeightIM','WeightRT']] = df.groupby(['O*NET-SOC Code','Occupation_VerbNouns'])[['Data Value IM','Data Value RT']].mean()
df = df.reset_index()
df_Tasks = df[['O*NET-SOC Code','Occupation_VerbNouns','count','count_total','WeightIM','WeightRT']].drop_duplicates()
df_Tasks = df_Tasks.loc[pd.notna(df_Tasks['WeightIM'])]

# Create Occupation-Task Vectors
dfIM,dfRT = df_Tasks[['O*NET-SOC Code','Occupation_VerbNouns','WeightIM']].copy(),df_Tasks[['O*NET-SOC Code','Occupation_VerbNouns','WeightRT']].copy()
dfIM = pd.pivot(dfIM,index='O*NET-SOC Code',columns='Occupation_VerbNouns',values='WeightIM').fillna(0)
dfRT = pd.pivot(dfRT,index='O*NET-SOC Code',columns='Occupation_VerbNouns',values='WeightRT').fillna(0)
MatIM = dfIM.to_numpy()
MatRT = dfRT.to_numpy()

# Construct "weighted" TF-IDF Scores
# TF(j,t) = weighted fraction of term t in document j
# IDF(t) = log(1+Num_Docs/sum([1 for j in range(Num_Docs) if t in doc]))
MatIM_TF,MatRT_TF = MatIM/MatIM.sum(axis=1)[:,np.newaxis],MatRT/MatRT.sum(axis=1)[:,np.newaxis]
MatIM_IDF,MatRT_IDF = np.log(1 + (MatIM.shape[1]/(MatIM>0).sum(axis=0))),np.log(1 + (MatRT.shape[1]/(MatRT>0).sum(axis=0)))
MatIM_TFIDF,MatRT_TFIDF = MatIM_TF*MatIM_IDF,MatRT_TF*MatRT_IDF

# Compute Similarity. Note that this is sort of wasteful since the occupation pairs (x,y) and (y,x) have the same similarity
# so each pair is computed twice; figure out a way to avoid this eventually. Finding all 4 combinations for similarity index 
# and RT/IM scale takes about 2 seconds an occupation. 

Occs = df_Tasks['O*NET-SOC Code'].unique() 
OccID = np.arange(len(Occs))

def CosineSimilarity(o1vec,o2vec):
    NR = (o1vec*o2vec).sum()
    DR1 = (o1vec**2).sum()
    DR2 = (o2vec**2).sum()
    return NR/np.sqrt(DR1*DR2)

def FractionSimilarity(o1vec,o2vec):
    return ((o1vec>0)*(o2vec>0)).sum()/o1vec.size

delCosSim = delayed(CosineSimilarity)
delFracSim = delayed(FractionSimilarity)
ExecutorFun = Parallel(n_jobs=8)

MatIMCosSim,MatIMFracSim,MatRTCosSim,MatRTFracSim = np.empty((len(Occs),len(Occs))),np.empty((len(Occs),len(Occs))),np.empty((len(Occs),len(Occs))),np.empty((len(Occs),len(Occs)))

for io in OccID:
    print(f'Occupation currently on: {Occs[io]}')
    t0 = time.time() 
    OutArrayIMCosSim = ExecutorFun([delCosSim(MatIM_TFIDF[io],MatIM_TFIDF[i]) for i in range(len(Occs))])
    t1 = time.time()
    OutArrayIMFracSim = ExecutorFun([delFracSim(MatIM_TFIDF[io],MatIM_TFIDF[i]) for i in range(len(Occs))])
    t2 = time.time()
    OutArrayRTCosSim = ExecutorFun([delCosSim(MatRT_TFIDF[io],MatRT_TFIDF[i]) for i in range(len(Occs))])
    t3 = time.time()
    OutArrayRTFracSim = ExecutorFun([delFracSim(MatRT_TFIDF[io],MatRT_TFIDF[i]) for i in range(len(Occs))])
    t4 = time.time()
    print(f'Time elapsed (s): IM,Cos = {t1-t0 :5.4f}, IM,Frac = {t2-t1 :5.4f}, RT,Cos = {t3-t2 :5.4f}, RT,Frac = {t4-t3 :5.4f}, total = {t4-t1 : 5.4f}')
    MatIMCosSim[io],MatIMFracSim[io] = np.array(OutArrayIMCosSim),np.array(OutArrayIMFracSim)
    MatRTCosSim[io],MatRTFracSim[io] = np.array(OutArrayRTCosSim),np.array(OutArrayRTFracSim)

# Construct DataFrames
dfIMCosSim = pd.DataFrame(data=MatIMCosSim,index=Occs,columns=Occs)
dfRTCosSim = pd.DataFrame(data=MatRTCosSim,index=Occs,columns=Occs)
dfIMFracSim = pd.DataFrame(data=MatIMFracSim,index=Occs,columns=Occs)
dfRTFracSim = pd.DataFrame(data=MatRTFracSim,index=Occs,columns=Occs)

dfIMCosSim.reset_index().to_csv('data/df_IMscale_cosine_similarity.csv')
dfRTCosSim.reset_index().to_csv('data/df_RTscale_cosine_similarity.csv')
dfIMFracSim.reset_index().to_csv('data/df_IMscale_fraction_similarity.csv')
dfRTFracSim.reset_index().to_csv('data/df_RTscale_fraction_similarity.csv')
