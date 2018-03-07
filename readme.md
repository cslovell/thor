# THOR: Text-enabled Humanitarian Operations in Real Time #
The THOR (Text-enabled Humanitarian Operations in Real-time) framework, funded under the DARPA LORELEI program, is designed to provide humanitarian and disaster relief planners superior situation awareness through visual and analytical data science.  
DARPA LORELEI program: https://www.darpa.mil/program/low-resource-languages-for-emergent-incidents  
USC Information Sciences Institute THOR homepage: http://usc-isi-i2.github.io/thor/  
Publication (Mayank Kejriwal, Haotian Zhang, et al) link: https://www.dropbox.com/s/0aytftod2p3973h/thor-text-enabled-final.pdf?dl=0  

# This Repo: Customized Document Feed Based on Multi-class Multi-label Classification #
This repo records the files related to Customized Document Feed Based on Multi-class Multi-label Classification, which is a subproblem belongs to the THOR project.  

## Objective ##
With natural disaster related massive documents come from various kinds of sources such as social network, online news and articles, etc., we aim to build a framework that can classify these documents into multiple themes, and feed them to related disaster relief organizations.  

A sample classification looks like the this (a document classified into four themes):  
![link1](https://s3-us-west-2.amazonaws.com/zhttestbucket/sample_doc_n_classification.png)  
  
This project is challenging and unique because:  
1. It is a multi-class multi-label classification task, which is much harder than binary or multi-class single-label classification tasks.  
2. The amount of data we received from DARPA is massive and unique. It contains millions of disaster-related documents that cannot be found anywhere else.  
3. We build advanced models to solve this challenge. They include Machine Learning, Deep Learning, Information Retrieval, Knowledge Graph, etc.   

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
















