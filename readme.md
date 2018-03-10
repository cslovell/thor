# THOR: Text-enabled Humanitarian Operations in Real Time #
The THOR (Text-enabled Humanitarian Operations in Real-time) framework, funded under the DARPA LORELEI program, is designed to provide humanitarian and disaster relief planners superior situation awareness through visual and analytical data science.  
DARPA LORELEI program: https://www.darpa.mil/program/low-resource-languages-for-emergent-incidents  
USC Information Sciences Institute THOR homepage: http://usc-isi-i2.github.io/thor/   

# This Repo: Customized Document Feed Based on Multi-class Multi-label Classification #
This repo records the files related to Customized Document Feed Based on Multi-class Multi-label Classification, which is a subproblem belongs to the THOR project. This is not the official repo for the project, just a personal development one.  

## Objective ##
With natural disaster related massive documents come from various kinds of sources such as social network, online news and articles, etc., we aim to build a framework that can classify these documents into multiple themes, and feed them to related disaster relief organizations.  

A sample classification looks like the this (a document classified into four themes):  
![link1](https://s3-us-west-2.amazonaws.com/zhttestbucket/sample_doc_n_classification.png)  
  
This project is challenging and unique because:  
1. It is a multi-class multi-label classification task, which is much harder than binary or multi-class single-label classification tasks.  
2. The amount of data we received from DARPA is massive and unique.    
3. We build advanced models to solve this challenge. They include Machine Learning, Information Retrieval, Knowledge Graph, etc.   

## Dataset ##
One sample document JSON object looks like this (text part is shortened):  
```
{  
   "lang":[  
      "en"
   ],
   "source_name":[  
      "UN Women"
   ],
   "glide":[  
      "EQ-2015-000048-NPL"
   ],
   "title":"A family rebuilds after the earthquake in Nepal",
   "country_location":[  
      83.94,
      28.25
   ],
   "text":"**In the aftermath of two devastating earthquakes in Nepal ... 5,400 households in the four hardest-hit districts.",
   "disaster_type":[  
      "Earthquake"
   ],
   "source_type":[  
      "International Organization"
   ],
   "theme":[  
      "Food and Nutrition",
      "Health",
      "Shelter and Non-Food Items",
      "Water Sanitation Hygiene"
   ],
   "href":"http://api.rwlabs.org/v1/reports/988536",
   "disaster_name":[  
      "Nepal: Earthquakes - Apr 2015"
   ],
   "country_name":"Nepal",
   "date_created":"2015-05-14T05:43:03+00:00",
   "id":"988536"
}
```
The main fields this repo interested in are "text" and "theme". Our goal is to learn from documents that labeled with themes, and predict themes for that are not labeled.  

## Working Pipeline ##
Here is a flow chart illustrates the working pipeline:  
![link1](https://s3-us-west-2.amazonaws.com/zhttestbucket/thor_flow.png)  

## Key Technology Used ##
1. For __Document Vector Embedding__, we use Facebook FastText (https://github.com/facebookresearch/fastText).  
2. For __TFIDF Vector Embeddings__, we use TFIDF package developed by myself (https://github.com/zhtpandog/tfidf). For extramely large-scale TFIDF classification, we also use sparse matrix in scipy package (https://github.com/scipy/scipy/blob/master/scipy/sparse/csr.py).  
3. For __Entity Extraction__, we use spaCy (https://github.com/explosion/spaCy).  
4. For __Random Walk Embeddings,__ we use DeepWalk (https://github.com/phanein/deepwalk) and NetworkX (https://github.com/networkx/networkx).  
5. For __Machine Learning__ models, we use scikit-learn (https://github.com/scikit-learn/scikit-learn).  
6. For various __Evaluation Metrics__, we use scikit-learn metrics module (https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/metrics) as well as custom metrics developed by myself such as NDCG.  

## Key ML Models Implemented ##
1. Supervised FastText document embedding classification.  
2. Unsupervised FastText document embedding + RidgeCV classification.  
3. Unsupervised FastText document embedding + Random Forest classification.  
4. TFIDF vectors + RidgeCV classification.  
5. TFIDF vectors + Random Forest classification.  
6. Random Walk on Bipartite Knowledge Graph Embeddings + RidgeCV classification.  
7. Random Walk on Bipartite Knowledge Graph Embeddings + Random Forest classification.  
(more to be added)  
  
















