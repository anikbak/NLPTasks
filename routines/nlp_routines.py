##########################################################
# ROUTINES TO PERFORM ANALYSIS OF TASKS AS IN WEBB (2020)
##########################################################

# Imports
import spacy, numpy as np, pandas as pd
from spacy.language import Language 
from nltk.corpus import wordnet as wn
from joblib import Parallel, delayed 

# Initialize NLP 
nlp = spacy.load('en_core_web_trf',disable=["ner","entity_linker","entity_ruler"])

# Routines for dependency parse and noun/verb extraction
def VerbNounPairs(doc):
    
    # Construct Verbs 
    verbs_all = [t for t in doc if t.pos_ == 'VERB']
    verbnounsD, verbnounsP = [],[] 

    # Search for Direct Objects
    for verb in verbs_all:

        # Find the first direct object on the subtree, if it exists.
        # Also add all conjunct nouns of the direct object.
        dobjs = [] 
        
        for tok in verb.subtree :
            if tok.dep_ == "dobj":
                dobjs.append(tok)
                verbnounsD.append((verb.lemma_,tok.text))
                ObjectTree = tok.subtree
                for o in ObjectTree:
                    if o.pos_ == "NOUN" and o.dep_ == "conj" and o.head.dep_ != "pobj" and o.dep_ != "pobj":
                        verbnounsD.append((verb.lemma_,o.text))
                break 
        
        # If no direct object was found, search for prepositional objects
        if dobjs == []:
            for tok in verb.subtree : 
                if tok.dep_ == "pobj" and tok.pos_ == 'NOUN' and tok.head.head.pos_ == "VERB":
                    verbnounsP.append((verb.lemma_,tok.head.text,tok.text))
    
    return verbnounsD, verbnounsP

def VerbNounPairs_Pipe(documentTuples):
    '''
    Given a list of tuples (doc,doc_index) of texts, construct verb and noun pairs.
    *******************************************************************************
    documentTuples: List of Tuples (text,doc_index)
    Output: Dict {k: [(v1,n1),(v2,n2),...,(vj,nj)]} where (v1,n1), (v2,n2), ..., (vj,nj) are tuples extracted from documentTuples[k]
    '''
    Tasks_dobj = {}
    for (doc,doc_index) in nlp.pipe(documentTuples,as_tuples=True,batch_size=200):
        Tasks_dobj[doc_index],_ = VerbNounPairs(doc)
    return Tasks_dobj

# Apply a Lemmatizer 
def word2hierarchy_word_minDepth(word,pos=wn.NOUN):

    # Start constructing WordNet Hierarchy: Use the first possible meaning (probably the best one)
    hierarchy = ['']
    hierarchy[0] = wn.synsets(word,pos=pos)[0]
    minDepth = hierarchy[0].min_depth()
    
    for i in range(1,minDepth):
        # For all later stages, use the minimum-depth synset 
        hypernyms = hierarchy[i-1].hypernyms()
        dists = [h.min_depth() for h in hypernyms]
        hypernym = [hypernyms[h] for h in range(len(hypernyms)) if dists[h]==min(dists)][0]
        hierarchy.append(hypernym)
    
    return hierarchy

def word2hierarchy_word_firsthypernym(word,pos=wn.NOUN):

    # Start constructing WordNet Hierarchy: Use the first possible meaning (probably the best one)
    hierarchy = ['']
    hierarchy[0] = wn.synsets(word,pos=pos)[0]
    minDepth = hierarchy[0].min_depth()
    
    for i in range(1,minDepth):
        # For all later stages, use the minimum-depth synset 
        hypernym = hierarchy[i-1].hypernyms()[0]
        hierarchy.append(hypernym)
    
    return hierarchy

def word2level(word_text,pos=wn.NOUN,level=5):
    '''
    Given a word, find its hierarchical ancestors based on synsets of increasing generality. 
    '''
    # Get correct synset (the first one possible)
    word_synsets = wn.synsets(word_text,pos=pos)
    if word_synsets == []:
        return word_text 
    else:
        word_current = word_synsets[0]
        minDepth = word_current.min_depth()
        if minDepth <= level:
            return word_current.lemmas()[0].name(),minDepth
        else:
            while minDepth > level:
                if word_current.hypernyms() == []:
                    return word_current.lemmas()[0].name(),minDepth
                else:
                    word_current = word_current.hypernyms()[0]
                    minDepth = word_current.min_depth() 
            return word_current.lemmas()[0].name(),minDepth

# Full Workflow for Task Extraction
def iterable2chunks(iterable,total_length,chunksize):
    return (iterable[pos:pos+chunksize] for pos in range(0,total_length,chunksize))

def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def flatten_dict(list_of_dicts):
    d = list_of_dicts[0]
    for du in list_of_dicts[1:]:
        d.update(du)
    return d

def Text2Tasks(df,TaskVar='Task',WordNetLevelMin=3,WordNetLevelMax=6,batch_size=100):
    '''
    Given a DataFrame df with a column TaskVar, construct variables containing different notions of tasks.
    ******************************************************************************************************
    df: pd.DataFrame. 
    TaskVar: pd.Series contained in pd.DataFrame. 
    '''

    df2 = df.copy()

    # Construct a List of Tasks
    print('Construct Task Lists and Preallocate')
    TasksList = [tuple(reversed(t)) for t in df[TaskVar].to_dict().items()]

    # Preallocate Dictionaries for Tasks and their Verb/Noun Pairs. 
    Tasks_dobj = {}

    # Create NLP Pipe to get tasks quickly 
    NLPDocs = nlp.pipe(TasksList,as_tuples=True,batch_size=batch_size)

    # Iterate over NLP Iterable 
    print('Iterate Over Tasks')
    for (doc,doc_index) in NLPDocs:
        Tasks_dobj[doc_index],_ = VerbNounPairs(doc)
    
    # Lemmatize verbs  
    Tasks_dobj = {k : [(Tasks_dobj[k][p][0].lemma_,Tasks_dobj[k][p][1]) for p in range(len(Tasks_dobj[k]))] for k in Tasks_dobj if Tasks_dobj[k] != []}    
     
    # Raw Tasks
    Tasks_dobj_df = pd.DataFrame.from_dict({k: str(Tasks_dobj[k]) for k in Tasks_dobj},orient='index',columns=['VerbNouns_dobj'])
    df2 = df2.merge(Tasks_dobj_df,how='outer',left_index=True,right_index=True,indicator='merge_verbnouns_raw')

    # Different Hierarchies 
    hierarchyLevels = np.arange(WordNetLevelMin,WordNetLevelMax+1)
    hierarchies_dobj = {}

    for l in hierarchyLevels:
        
        # Store in Hierarchy Dictionaries
        hierarchies_dobj[l] = {k : str([(Tasks_dobj[k][p][0],word2level(Tasks_dobj[k][p][1].text,level=l)[0]) for p in range(len(Tasks_dobj[k]))]) for k in Tasks_dobj if Tasks_dobj[k] != []}

        # Construct DataFrames
        Tasks_dobj_df = pd.DataFrame.from_dict(hierarchies_dobj[l],orient='index',columns=['VerbNouns_dobj_NounLevel_'+str(l)])
        df2 = df2.merge(Tasks_dobj_df,how='outer',left_index=True,right_index=True,indicator='merge_verbnouns_level_'+str(l))

    return Tasks_dobj, hierarchies_dobj, df2 

def Text2TasksDict_Parallelized(df,TaskVar='Task',chunksize=1000,joblib_n_jobs=-1,joblib_parallel_backend='loky',joblib_parallel_prefer='processes'):
    
    # Get Tuples
    TasksList = [tuple(reversed(t)) for t in df[TaskVar].to_dict().items()]

    # Parallel and Delayed Objects
    executor = Parallel(n_jobs=joblib_n_jobs,backend=joblib_parallel_backend,prefer=joblib_parallel_prefer,verbose=20)
    do = delayed(VerbNounPairs_Pipe)
    
    # Construct Tasks!
    tasks = (do(chunk) for chunk in iterable2chunks(TasksList,len(TasksList),chunksize))
    result = executor(tasks)
    return flatten_dict(result)

def Text2Tasks_Parallelized(df,TaskVar='Task',chunksize=1000,joblib_n_jobs=-1,joblib_parallel_backend='loky',joblib_parallel_prefer='processes',WordNetLevelMin=3,WordNetLevelMax=6):

    # Preallocate
    df2 = df.copy()

    # Get Dictionary of Tuples
    if len(df.index) <= 2*chunksize: 
        joblib_n_jobs = 1

    Tasks_dobj = Text2TasksDict_Parallelized(df,TaskVar=TaskVar,
                                                chunksize=chunksize,
                                                joblib_n_jobs=joblib_n_jobs,
                                                joblib_parallel_backend=joblib_parallel_backend,
                                                joblib_parallel_prefer=joblib_parallel_prefer)
     
    # Raw Tasks
    Tasks_dobj_df = pd.DataFrame.from_dict({k: str(Tasks_dobj[k]) for k in Tasks_dobj},orient='index',columns=['VerbNouns_dobj'])
    df2 = df2.merge(Tasks_dobj_df,how='outer',left_index=True,right_index=True,indicator='merge_verbnouns_raw')

    # Different Hierarchies 
    hierarchyLevels = np.arange(WordNetLevelMin,WordNetLevelMax+1)
    hierarchies_dobj = {}

    for l in hierarchyLevels:
        
        # Store in Hierarchy Dictionaries
        hierarchies_dobj[l] = {k : str([(Tasks_dobj[k][p][0],word2level(Tasks_dobj[k][p][1],level=l)[0]) for p in range(len(Tasks_dobj[k]))]) for k in Tasks_dobj if Tasks_dobj[k] != []}

        # Construct DataFrames
        Tasks_dobj_df = pd.DataFrame.from_dict(hierarchies_dobj[l],orient='index',columns=['VerbNouns_dobj_NounLevel_'+str(l)])
        df2 = df2.merge(Tasks_dobj_df,how='outer',left_index=True,right_index=True,indicator='merge_verbnouns_level_'+str(l))

    return Tasks_dobj, hierarchies_dobj, df2
     
# For analysis of tasks and most common tasks by year, construct dataset at the Occupation by Extracted Task level 
def DataOccTaskLevel(df,OccVar="O*NET-SOC Code",otherVars=['Task ID','Task Type','Incumbents Responding','Domain Source'],WordNetLevel=None,TaskVar='Task',NewTaskVar='Occupation_VerbNouns'):
    '''
    Construct a dataset of occupations and tasks. 
    *********************************************
    df: DataFrame containing at least the columns OccVar, TaskVar, VerbNouns_dobj_NounLevel_+str(WordNetLevel) 
    '''
    # Variable
    if WordNetLevel is None:
        TaskTupleVar = 'VerbNouns_dobj'
    else:
        TaskTupleVar = 'VerbNouns_dobj_NounLevel_'+str(WordNetLevel)

    # Split up Tasks
    df[TaskTupleVar+'_list'] = df[TaskTupleVar].apply(eval)
    df_Tasks_AsCols = df[TaskTupleVar+'_list'].apply(pd.Series)
    MaxNumTasks = len(df_Tasks_AsCols.columns)
    df_Tasks_AsCols = df_Tasks_AsCols.rename(columns={c:'Task_'+str(c) for c in df_Tasks_AsCols.columns})
    df = df.merge(df_Tasks_AsCols,left_index=True,right_index=True,indicator='_m_TaskCols',how='outer')

    # Reshape and build a new DataFrame to prepare for collapse 
    df_ = pd.DataFrame(columns=[OccVar,TaskVar,TaskTupleVar,NewTaskVar,'OrigTaskIndex']+otherVars)
    
    for i in range(MaxNumTasks):
        dfTemp = df[[OccVar,TaskVar,TaskTupleVar,TaskTupleVar+'_list','Task_'+str(i)]+otherVars].copy()
        dfTemp = dfTemp.loc[pd.notna(dfTemp['Task_'+str(i)])]
        if len(dfTemp.index)>0:
            dfTemp = dfTemp.rename(columns={'Task_'+str(i) : NewTaskVar})
            dfTemp['OrigTaskIndex'] = i
            df_ = df_.append(dfTemp,sort=False) 

    return df_

