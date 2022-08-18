########################################################################################################
# STUDY TASKS IN O*NET DATASET
########################################################################################################

# Imports
from routines.nlp_routines import Text2Tasks_Parallelized, Text2TasksDict_Parallelized
import pandas as pd

# Load Datasets
df = pd.read_csv('data_raw/onet/Task Statements.txt', sep='\t')
df = df.sort_values(['O*NET-SOC Code','Task Type','Task ID','Task'])
df = df.reset_index() 

# Add "They" to each task, virtually ensuring that the NLP will recognize the first word of each task statement as a verb. 
df['Task'] = 'They '+df['Task'].str.lower() 

# Extract Tasks
vn = Text2TasksDict_Parallelized(df,TaskVar='Task',joblib_parallel_backend='threading',chunksize=200)

df.to_csv('data/data_onet_task_verbnoun_pairs.csv')