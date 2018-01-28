# from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MultiLabelBinarizer
import json
import os
import codecs
import operator
from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid



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


def custom_ndcg_grid_search_cv(model, X, y, param_dict, save_matrix=True, matrix_id="", track_folder="track2/"):

    assert len(X) == len(y), "X and y dimension not match!"

    param_list = list(ParameterGrid(param_dict))
    num_samples = len(X)
    num_param_comb = len(param_list) # number of parameter combinations, cross product of param list
    result_matrix = [[0 for j in range(num_samples)] for i in range(num_param_comb)]
    count_p = 0

    for p_ind, param in enumerate(param_list):

        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X):
            X_train = [X[i] for i in train_index]
            y_train = [y[i] for i in train_index]
            X_test = [X[i] for i in test_index]
            y_test = [y[i] for i in test_index]

            clf = model(**param)
            clf.fit(X_train, y_train)
            y_predrid = clf.predict_proba(X_test)

            # for random forest
            y_predrid_format = [[0 for item2 in range(len(y_predrid))] for item1 in
                                range(len(y_predrid[0]))]  # 2992 * 19
            for class_idx, class_list in enumerate(y_predrid):  # class_idx is 0-18, class list's size is 2992 * 2
                for sample_idx, i in enumerate(class_list):
                    y_predrid_format[sample_idx][class_idx] = i[1]

            for idx, t_idx in enumerate(test_index):
                ndcg = custom_ndcg_score(y_test[idx], y_predrid_format[idx])
                result_matrix[p_ind][t_idx] = ndcg

        count_p += 1
        with open(track_folder + str(count_p) + "_" + str(len(param_list)), "w") as fi:
            fi.write("ok")

    if save_matrix:
        with open("ndcg_grid_matrix_" + matrix_id + ".json", "w") as f:
            f.write(json.dumps(result_matrix))

    # find optimum param comb based on avg ndcg
    param_n_avg_ndcg = []
    best_ndcg = 0
    best_param = None
    for idx, res in result_matrix:
        avg_ndcg = sum(res) / len(res)
        param_n_avg_ndcg.append({"params": param_list[idx], "avg_ndcg": avg_ndcg})
        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            best_param = param_list[idx]

    return (best_param, best_ndcg, param_n_avg_ndcg)


def input_process(X_filename, y_filename):
    # X_filename is what received from fastText, y_filename is what we prepared
    # perpare X
    f = open(X_filename, "r") # "raw.txt"
    X = []
    for line in f:
        tmp = []
        ls = line.strip().split(" ")
        for i in [-j for j in range(100, 0, -1)]:
            tmp.append(float(ls[i]))
        X.append(tmp)

    # transform the y labels into binary
    raw_y = json.loads(open(y_filename).read())
    mlb = MultiLabelBinarizer()
    bin_y_dev = mlb.fit_transform(raw_y)
    y = []
    for i in bin_y_dev:
        y.append(list(i))

    return (X, y)


def train_ft(param_dict, input_filename, X_filename):

    # cmd_p = [fasttext_dir, "/fasttext skipgram -input data.unsup_all.txt -output model"]
    cmd_p = [fasttext_dir, "/fasttext skipgram -input " + input_filename + " -output model"]

    for p_name, p_val in param_dict.items():
        cmd_p.append(" -")
        cmd_p.append(p_name)
        cmd_p.append(" ")
        cmd_p.append(str(p_val))

    cmd_p = "".join(cmd_p)
    os.system(cmd_p)

    os.system(fasttext_dir + "/fasttext print-sentence-vectors model.bin < data.unsup_all.txt > " + X_filename)


y_filename = "y_rawlabels_dev.json"
fasttext_dir = "/home/entitylinking/fastText"
ft_param_dict_optimal = {"wordNgrams": 3, "lr": 0.05, "ws": 8} # sample
X_filename = "fastText_best_result.txt" # fastText best result file
rf_parameters = {"n_estimators": [10, 20, 30, 40, 50], "criterion": ["gini", "entropy"],
                  "max_features": ["auto", "sqrt", "log2", None], "n_jobs": [-1]}

train_ft(ft_param_dict_optimal, "data.unsup_all.txt", X_filename)
(X, y) = input_process(X_filename, y_filename)
(best_param, best_ndcg, param_n_avg_ndcg) = custom_ndcg_grid_search_cv(RandomForestClassifier, X, y, rf_parameters, matrix_id="1")

ret = {"best_param": best_param, "best_ndcg": best_ndcg, "param_n_avg_ndcg": param_n_avg_ndcg}

with open("grid_cv_summary.json", "w") as f:
    f.write(json.dumps(ret))





# from sklearn.metrics import make_scorer
# from sklearn.model_selection import GridSearchCV
# ndcg_scorer = make_scorer(custom_ndcg_score, needs_proba=True)
# parameters = {"n_estimators": [10, 20, 30, 40, 50], "criterion": ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2", None], "n_jobs": [-1]}
# rfc = RandomForestClassifier()
# clf = GridSearchCV(rfc, param_grid=parameters, scoring=ndcg_scorer, n_jobs=4)
# clf.fit(vec_X_dev, bin_y_dev_list)
# clf.predict_proba(vec_X_dev)