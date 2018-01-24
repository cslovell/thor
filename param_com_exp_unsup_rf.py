# from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MultiLabelBinarizer
import json
import os
import codecs
import operator
from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier


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

def get_param_comb_list(param):
    # see if there is any proprocess commands
    prepro = []
    if "preprocess" in param:
        prepro = param["preprocess"]
        del param["preprocess"]

    # get param combinations
    num_param = len(param)

    if prepro:
        param_tokens = ["a"] + [chr(ord("b") + i) for i in range(num_param)]
    else:
        param_tokens = [chr(ord("a") + i) for i in range(num_param)]

    comb_cmd = ["[", "(" + ",".join(param_tokens) + ")"]
    param_items = param.items()
    param_cates = []
    if prepro:
        param_cates.append("preprocess")
        comb_cmd.append(" for ")
        comb_cmd.append(param_tokens[0])
        comb_cmd.append(" in ")
        comb_cmd.append(str(prepro))

        for idx, content in enumerate(param_items):
            param_cates.append(content[0])
            comb_cmd.append(" for ")
            comb_cmd.append(param_tokens[idx + 1])
            comb_cmd.append(" in ")
            comb_cmd.append(str(content[1]))
    else:
        for idx, content in enumerate(param_items):
            param_cates.append(content[0])
            comb_cmd.append(" for ")
            comb_cmd.append(param_tokens[idx])
            comb_cmd.append(" in ")
            comb_cmd.append(str(content[1]))

    comb_cmd.append("]")
    comb_cmd = "".join(comb_cmd)

    param_comb_list = eval(comb_cmd)

    return (param_comb_list, param_cates)


###
config = json.loads(open("config.json").read())

param_dict = config["param_dict"]
num_labels = config["num_labels"]
fasttext_dir = config["fasttext_dir"]
y_data = config["y_data"]

y = json.loads(open(y_data).read())
(param_comb_list, param_cates) = get_param_comb_list(param_dict)

result_matrix = [[0 for j in y] for i in param_comb_list]

total_p = len(param_comb_list)
count_p = 0
for p_ind, p in enumerate(param_comb_list):

    cmd_p = [fasttext_dir, "/fasttext skipgram -input data.unsup_all.txt -output model"]

    for idx, par in enumerate(param_cates):
        cmd_p.append(" -")
        cmd_p.append(param_cates[idx])
        cmd_p.append(" ")
        cmd_p.append(str(p[idx]))

    # get the model.bin
    cmd_p = "".join(cmd_p)
    os.system(cmd_p)

    # compute vector representations for docs
    os.system(fasttext_dir + "/fasttext print-sentence-vectors model.bin < data.unsup_all.txt > raw.txt")

    # after having the text+vector, get the vector
    f = open("raw.txt", "r")
    cnt = 0
    result = []
    for line in f:
        tmp = []
        ls = line.strip().split(" ")
        for i in [-j for j in range(100, 0, -1)]:
            tmp.append(float(ls[i]))
        result.append(tmp)

    # transform the y labels into binary
    vec_X_dev = result

    mlb = MultiLabelBinarizer()
    bin_y_dev = mlb.fit_transform(y)
    bin_y_dev_list = []
    for i in bin_y_dev:
        bin_y_dev_list.append(list(i))

    #
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(vec_X_dev):
        X_train = [vec_X_dev[i] for i in train_index]
        y_train = [bin_y_dev_list[i] for i in train_index]
        X_test = [vec_X_dev[i] for i in test_index]
        y_test = [bin_y_dev_list[i] for i in test_index]

        # model kernel here
        # rid = RidgeCV()
        # ridModel = rid.fit(X_train, y_train)
        # y_predrid = rid.predict(X_test)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_predrid = clf.predict(X_test)

        temp_count = 0
        for i in test_index:
            ndcg = custom_ndcg_score(y_test[temp_count], y_predrid[temp_count])
            result_matrix[p_ind][i] = ndcg
            temp_count += 1

    count_p += 1
    print "ok"
    # break
    with open("track1/" + str(count_p), "w") as fi:
        fi.write("ok")

with open("unsup_matrix_forest.json", "w") as f:
    f.write(json.dumps(result_matrix))

