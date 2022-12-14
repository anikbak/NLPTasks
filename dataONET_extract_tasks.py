########################################################################################################
# CONSTRUCT NOUN/VERB TASKS IN O*NET DATASET
########################################################################################################

# Imports
from routines.nlp_routines import Text2Tasks_Parallelized, DataOccTaskLevel
import pandas as pd

# Load Datasets
'''
This Data is from O*NET version 26.3, downloaded on 8/16/2022. 

Task Statements.txt contains 19,258 unique task statements. 
'''

df = pd.read_csv('data_raw/onet/Task Statements.txt', sep='\t')
df = df.sort_values(['O*NET-SOC Code','Task Type','Task ID','Task'])
df = df.reset_index() 

# Add "They" to each task, virtually ensuring that the NLP will recognize the first word of each task statement as a verb. 
df['Task'] = 'They '+df['Task'].str.lower() 

# Extract Tasks
_,_,df_with_tasks = Text2Tasks_Parallelized(df,TaskVar='Task',joblib_parallel_backend='threading',joblib_parallel_prefer='processes',chunksize=200)
df_with_tasks.to_csv('data/data_onet_with_tasks.csv')

# Construct Task Data
df_with_tasks_ = DataOccTaskLevel(df_with_tasks)
df_with_tasks_ = df_with_tasks_.sort_values(['O*NET-SOC Code','Task Type','Task ID','Task','OrigTaskIndex'])
df_with_tasks_.to_csv('data/data_onet_task_verbnoun_pairs.csv')

'''
The code identifies an average of 3.54 (verb,noun) pairs per occupational task. 
'''

# Construct Generality Level-4 Task Data
df_with_tasks_4 = DataOccTaskLevel(df_with_tasks,WordNetLevel=4)
df_with_tasks_4 = df_with_tasks_4.sort_values(['O*NET-SOC Code','Task Type','Task ID','Task','OrigTaskIndex'])
df_with_tasks_4.to_csv('data/data_onet_task_nounlevel4_verbnoun_pairs.csv')
