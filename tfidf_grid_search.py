from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
import json
import operator
from sklearn.model_selection import KFold
import numpy as np
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

bin_y_dev_ready = []
for i in id_list:
    bin_y_dev_ready.append(id_and_bin_y_dev[i])

# transform dict_list to sparse
v = DictVectorizer()
X = v.fit_transform(dict_list)

###
def custom_ndcg_grid_search_cv(model, X, y, param_dict, save_matrix=True, matrix_id="", track_folder="track3/"):

    param_list = list(ParameterGrid(param_dict))
    num_samples = len(y)
    num_param_comb = len(param_list) # number of parameter combinations, cross product of param list
    result_matrix = [[0 for j in range(num_samples)] for i in range(num_param_comb)]
    count_p = 0

    best_result_record = [[] for j in y]  # for single iteration

    for p_ind, param in enumerate(param_list):

        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X):
            X_train = vstack([X[i] for i in train_index], "csr")
            y_train = [y[i] for i in train_index]
            X_test = vstack([X[i] for i in test_index], "csr")
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
                best_result_record[t_idx] = list(y_predrid_format[idx])

        count_p += 1
        with open(track_folder + str(count_p) + "_" + str(len(param_list)), "w") as fi:
            fi.write("ok")

    with open("detail_proba_" + matrix_id + ".json", "w") as f:
        f.write(json.dumps(best_result_record))

    if save_matrix:
        with open("ndcg_grid_matrix_" + matrix_id + ".json", "w") as f:
            f.write(json.dumps(result_matrix))

    # find optimum param comb based on avg ndcg
    param_n_avg_ndcg = []
    best_ndcg = 0
    best_param = None
    for idx, res in enumerate(result_matrix):
        avg_ndcg = sum(res) / len(res)
        param_n_avg_ndcg.append({"params": param_list[idx], "avg_ndcg": avg_ndcg})
        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            best_param = param_list[idx]

    return (best_param, best_ndcg, param_n_avg_ndcg)

ID = "rf_tfidf_1000_best"
rf_parameters = {"n_estimators": [1000], "criterion": ["gini"],
                 "max_features": ["auto"], "n_jobs": [-1], "min_samples_leaf": [0.00001]}

# rf_parameters = {"n_estimators": [10, 20, 50, 100, 500], "criterion": ["gini", "entropy"],
#                  "max_features": ["auto", "sqrt", "log2"], "n_jobs": [-1], "min_samples_leaf": [0.00001, 0.0001, 0.001]}

(best_param, best_ndcg, param_n_avg_ndcg) = custom_ndcg_grid_search_cv(RandomForestClassifier, X, bin_y_dev_ready, rf_parameters, matrix_id=ID)

ret = {"best_param": best_param, "best_ndcg": best_ndcg, "param_n_avg_ndcg": param_n_avg_ndcg}

with open("grid_cv_summary_" + ID + ".json", "w") as f:
    f.write(json.dumps(ret))