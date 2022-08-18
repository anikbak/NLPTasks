# NLPTasks

## Description  
Codes to apply an extension of the method in Webb (2020) to extract Tasks from text describing an occupation or a technology. A task is defined as a (verb,noun) pair where the noun is the direct object, or a conjunct of the direct object, of the verb.

## Steps Involved

* Step 1: Using SpaCy's DependencyParser and the #en_core_web_sm# model, add dependency and parts-of-speech tags to all tokens in a text. 
* Step 2: For all words identified as verbs by the POS Tagger, identify the first direct object that lies on the verb's subtree constructed by the .head relationship. 