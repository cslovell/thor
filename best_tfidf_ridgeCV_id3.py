from sklearn.feature_extraction import DictVectorizer
import json
import operator
import requests
import re
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.sparse import vstack

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

id_list = json.loads(open("id_list.json").read())
dict_list = json.loads(open("dict_list.json").read())
id_and_bin_y_dev = json.loads(open("id_and_bin_y_dev.json").read())
class_labels = json.loads(open("class_labels_19.json").read())

bin_y_dev_ready = []
for i in id_list:
    bin_y_dev_ready.append(id_and_bin_y_dev[i])

# transform dict_list to sparse
v = DictVectorizer()
X = v.fit_transform(dict_list)

# data ready, X is X, y is bin_y_dev_ready
kf = KFold(n_splits=10)
result_vec = [0 for i in range(29912)]
count = 0

with open("track1/" + "ready_to_start" + str(count), "w") as fi:
    fi.write("ok")

best_result_record = [[] for j in y]

for train_index, test_index in kf.split(X):
    X_train = vstack([X[i] for i in train_index], "csr")
    y_train = [bin_y_dev_ready[i] for i in train_index]
    X_test = vstack([X[i] for i in test_index], "csr")
    y_test = [bin_y_dev_ready[i] for i in test_index]

    rid = RidgeCV()
    ridModel = rid.fit(X_train, y_train)
    y_predrid = rid.predict(X_test)

    temp_count = 0
    for i in test_index:
        ndcg = custom_ndcg_score(y_test[temp_count], y_predrid[temp_count])
        result_vec[i] = ndcg
        best_result_record[i] = list(y_predrid[temp_count])
        temp_count += 1

    count += 1
    with open("track1/" + "finish_" + str(count), "w") as fi:
    fi.write("ok")


best_result_record_ordered = []
for i in best_result_record:
    temp = []
    for j in range(19):
        temp.append([class_labels[j], i[j]])
    temp.sort(key=lambda x:x[1], reverse=True)
    best_result_record_ordered.append(temp)

with open("result_lbl_n_prob_best_tfidf_RidgeCV_id3.json", "w") as f: ##need change
    f.write(json.dumps(best_result_record_ordered))

with open("result_ndcg_best_tfidf_RidgeCV_id3.json", "w") as f: ##need change
    f.write(json.dumps(result_vec))

with open("track1/" + "finish_all", "w") as fi:
    fi.write("ok")