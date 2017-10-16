import math
import numpy as np
import jsonlines
import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

class tfidf2(object):
  def __init__(self):
    self.documents = {} # {doc:{word:tfidf}} after call prep(), contains each doc name and its doc dict, the doc dict is {word:tfidf}
    self.corpus_dict = {} # {word:df}, contains all the words, and document frequency (df)
    self.idf = {} # idf
    self.num_docs = 0 # number of documents
    self.prepStatus = False

  def addDocument(self, doc_name, list_of_words):
    '''
    Add document one by one
    :param doc_name: document name, or uuid
    :param list_of_words: word list correspond to the doc name
    :return: void
    '''
    ## compute tf (doc dict is the dict of the single doc)
    doc_dict = {}
    for w in list_of_words:
      doc_dict[w] = doc_dict.get(w, 0.0) + 1.0 # if the word w exists, plus 1 to its value; if not exists, make its value 1

    # normalizing the doc dict (creating tf score)
    length = float(len(list_of_words))
    for k in doc_dict:
      doc_dict[k] = doc_dict[k] / length

    # add the normalized document and its tf score to the corpus
    self.documents[doc_name] = doc_dict
    ## finish the work on tf

    # make change to the global df
    for w in set(list_of_words):
      self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0 # count each word's to the whole corpus contribution only once

  def prep(self):
    '''
    Prepare the tfidf value for each doc in corpus.
    :return: void
    '''
    # creating idf dict
    self.num_docs = len(self.documents)
    for i, j in self.corpus_dict.items():
      self.idf[i] = math.log(self.num_docs / self.corpus_dict[i])

    # computing tfidf for each document
    for doc in self.documents:
      for i in self.documents[doc]: # i is word
        self.documents[doc][i] *= self.idf[i]
    self.prepStatus = True

  def get_tfidf(self):
      if self.prepStatus:
          return self.documents
      else:
          print "not prepared"



class MTokenizer(object):

    @staticmethod
    def tokenize_string(string):
       """
       I designed this method to be used independently of an obj/field. If this is the case, call _tokenize_field.
       It's more robust.
       :param string: e.g. 'salt lake city'
       :return: list of tokens
       """
       list_of_sentences = list()
       tmp = list()
       tmp.append(string)
       k = list()
       k.append(tmp)
       # print k
       list_of_sentences += k  # we are assuming this is a unicode/string

       word_tokens = list()
       for sentences in list_of_sentences:
           # print sentences
           for sentence in sentences:
               for s in sent_tokenize(sentence):
                   word_tokens += word_tokenize(s)

       return word_tokens

table = tfidf2()
with jsonlines.open("reliefweb_corpus_raw_20160331_eng_duprm.jsonl") as reader:
    count = 0
    for obj in reader:
        table.addDocument(obj["id"], MTokenizer.tokenize_string(obj["text"]))
        count += 1
        if count % 1000 == 0:
            with open("track1/" + str(count), "w") as fi:
                fi.write("ok")

with open("track1/" + "finish_load", "w") as fi:
    fi.write("ok")

table.prep()

with open("track1/" + "finish_perp", "w") as fi:
    fi.write("ok")

res = table.get_tfidf()

with open("track1/" + "finish_len_" + str(len(res)), "w") as fi:
    fi.write("ok")

with open("tfidf_all.json", "w") as f:
    f.write(json.dumps(res))