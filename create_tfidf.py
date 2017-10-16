import math
import numpy as np
import jsonlines
import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import os
import codecs
import operator
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV


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

    ans = self.documents

    return ans



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



def custom_ndcg_score(y, y_pred):

    # get index of all correct answers, and get smallest number among them
    correct_index = set()
    smallest = 1
    for idx, label in enumerate(y):
        if label == 1:
            correct_index.add(idx)
            if y_pred[idx] < smallest:
                smallest = y_pred[idx]

    # include value for correct indices and valued larger than smallest correct
    index_and_val = {}
    for idx, val in enumerate(y_pred):
        if idx in correct_index:
            index_and_val[idx] = y_pred[idx]
        elif val > smallest:
            index_and_val[idx] = y_pred[idx]

    index_and_val = sorted(index_and_val.items(), key=operator.itemgetter(1), reverse=True)

    #
    pred_list = []
    for idx, val in index_and_val:
        if idx in correct_index:
            pred_list.append(1)
        else:
            pred_list.append(0)

    # idcg
    idcg = 0
    for idx in range(len(correct_index)):
        idcg += 1 / np.log2(idx + 1 + 1)

    # dcg
    dcg = 0
    for idx, score in enumerate(pred_list):
        dcg += score / np.log2(idx + 1 + 1)

    return dcg / idcg



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

res = table.prep()

with open("track1/" + "finish_len_" + str(len(res)), "w") as fi:
    fi.write("ok")


# with open("tfidf_all.json", "w") as f:
#     f.write(json.dumps(res)) # 423973

id_dev = json.loads(open("ids_raw_dev.json").read())
bin_y_dev_list = json.loads(open("bin_y_dev_list.json").read())

dict_list = []
id_fail = []
for i in id_dev:
    try:
        dict_list.append(res[i])
    except:
        id_fail.append(i)

X = []
if len(dict_list) == 29912:
    with open("track1/" + "length_29912_normal", "w") as fi:
        fi.write("ok")

    # create sparse matrix
    v = DictVectorizer()
    X = v.fit_transform(dict_list)

    # X: X, y = bin_y_dev_list
    kf = KFold(n_splits=10)
    result_vec = [0 for i in range(29912)]
    for train_index, test_index in kf.split(X):
        X_train = [X[i] for i in train_index]
        y_train = [bin_y_dev_list[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        y_test = [bin_y_dev_list[i] for i in test_index]

        rid = RidgeCV()
        ridModel = rid.fit(X_train, y_train)
        y_predrid = rid.predict(X_test)

        temp_count = 0
        for i in test_index:
            ndcg = custom_ndcg_score(y_test[temp_count], y_predrid[temp_count])
            result_vec[i] = ndcg
            temp_count += 1

    with open("result_vec.json", "w") as fi:
        fi.write(result_vec)
else:
    with open("track1/" + "something_wrong", "w") as fi:
        fi.write("ok")

with open("track1/" + "PROG_TO_END", "w") as fi:
    fi.write("ok")





# with open("tfidf_all.json", "w") as f:
#     f.write(json.dumps(res))

