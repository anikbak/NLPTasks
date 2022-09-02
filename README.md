# NLPTasks

## Description  
Codes to apply an extension of the method in Webb (2020) to extract Tasks from text describing an occupation or a technology. A task is defined as a (verb,noun) pair where the noun is the direct object, or a conjunct of the direct object, of the verb.

## Steps Involved

* Step 1: Using SpaCy's DependencyParser and the ``en_core_web_trf`` model, add dependency and parts-of-speech tags to all tokens in a text. 
* Step 2: For all words identified as verbs by the POS Tagger, identify the first direct object that lies on the verb's subtree defined by the .head relationship. 
* Step 3: Use SpaCy's Lemmatizer to lemmatize verbs. 
* Step 4: Use NLTK's WordNet Corpus to translate nouns to a common level of generality (WordNet Level 4). 

## Similarity Matrices

Once verb-noun tasks have been extracted from the O\*NET Data, we can do cool stuff with the vector representation of occupations in the verb-noun space. Here's a natural one: How "similar" are different occupations? 

To answer this, here's one approach based on the idea of TF-IDF Measures. Let $t=1,2,\dots,T$ index verb-noun tasks and $o=1,2,\dots,O$ index occupations. Let $w(o) = [ w(o,1),w(o,2),\dots,w(o,T) ] \in \mathbb{R}^T$ be the vector of loadings on tasks $t=1,2,\dots,T$ for occupation $o$. These loadings are constructed as follows. 

* Pick an appropriate scale from O\*NET for each raw task associated with each occupation. There are two scales, importance and relevance. Let $w(o,\hat{t})$ be the scale value of raw task $\hat{t}$ for occupation $o$. If $w(o,\hat{t})>0$ I will use the notation $\hat{t}\in o$.

* After the decomposition of raw task $\hat{t}$ into verb-noun tasks, I will use the notation $t\in \hat{t}$ if verb-noun task $t$ was extracted from raw task $\hat{t}$. The notation $t\in o$ will indicate that $t\in\hat{t}$ for at least one $\hat{t}\in o$.

* Construct a scale value for each verb-noun task by taking the mean scale value across all raw tasks within an occupation mentioning that verb-noun task. That is, for verb-noun task $t$, 

$$ w(o,t) = \frac{1}{N(o,t)}\sum_{\hat{t} \; : \; t \in \hat{t}, \; \hat{t} \in o} w\left( o,\hat{t} \right) $$

where $N(o,t) = \sum_{\hat{t} : t \in \hat{t}, \hat{t} \in o} \mathbf{1}$.

* Construct the Weighted Term Frequency of task $t$ for occupation $o$ using the formula

$$ W_{TF}(o,t) = \frac{w(o,t)}{\sum_{t\in o^{\prime}} w(o^{\prime},t)} $$

* Construct the Inverse Document Frequency of task $t$ using the formula 

$$ W_{IDF}(t) = \log\left(1 + \frac{O}{ \sum_{o=1,\dots,O} \mathbf{1}\left(t \in o \right)} \right) $$

* Construct the TF-IDF Scores $$w(o,t) = W_{TF}(o,t)\times W_{IDF}(t)$$. 

The TF-IDF Scores give us a vector representation of each occupation in terms of the space of extracted tasks. We can now use these scores to compute the similarity between two occupations. I compute two different measures:

* The Cosine Similarity, defined by 
$$CosSim(o,o^{prime}) = \frac{w(o)\cdot w(o^{\prime})}{\vert w(o)\vert\times\vert w(o^{\prime})\vert}$$

where $$\vert w(o) \vert = \sqrt{\sum_{t=1}^T w(o,t)^2}$$ and $$w(o)\cdot w(o^{\prime}) = \sum_{t=0}^T w(o,t)w(o^{\prime},t)$$.

* The Fraction Similarity, defined by 
$$FracSim(o,o^{prime}) = \frac{\mathbf{1}(w(o)>0)\cdot\mathbf{1}(w(o^{\prime})>0)}{T}$$
