########################################################################################################
# STUDY TASKS IN O*NET DATASET
########################################################################################################

# Imports
from routines.nlp_routines import Text2Tasks_Parallelized, DataOccTaskLevel
import pandas as pd

# Load Datasets
df = pd.read_csv('data_raw/onet/Task Statements.txt', sep='\t')
df = df.sort_values(['O*NET-SOC Code','Task Type','Task ID','Task'])
df = df.reset_index() 

# Add "They" to each task, virtually ensuring that the NLP will recognize the first word of each task statement as a verb. 
df['Task'] = 'They '+df['Task'].str.lower() 

# Extract Tasks
_,_,df_with_tasks = Text2Tasks_Parallelized(df.iloc[:100],TaskVar='Task',joblib_parallel_backend='threading',joblib_parallel_prefer='processes',chunksize=200)
df_with_tasks = DataOccTaskLevel(df_with_tasks)
df_with_tasks = df_with_tasks.sort_values(['O*NET-SOC Code','Task Type','Task ID','Task','OrigTaskIndex'])
df_with_tasks.to_csv('data/data_onet_task_verbnoun_pairs.csv')

# Construct DataFrame at the Task Level