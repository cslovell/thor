import json
import numpy as np
import matplotlib.pyplot as plt

sup_matrix = json.loads(open("result_matrix_correct.json").read()) # avg value of each param comb (70 in total): avg: 0.7930, min: 0.6551, max: 0.9062, std: 0.0930
unsup_matrix = json.loads(open("unsup_matrix.json").read()) # avg value of each param comb (70 in total): avg: 0.8773, min: 0.8679, max: 0.9008, std: 0.0141
tfidf_vec = json.loads(open("result_vec.json").read()) # tfidf average: 0.9209
unsup_matrix_forest = json.loads(open("unsup_matrix_forest.json").read()) # avg: 0.9434, min: 0.9401, max: 0.9492, std: 0.0027

param_comb_list = json.loads(open("param_comb_list.json").read())
id_list = json.loads(open("ids_raw_dev.json").read())
labels_list = json.loads(open("y_rawlabels_dev.json").read())

## 1.avg
# average ndcg for each param set
x_axis = [i for i in range(1,71)]
avg_param_comb_sup_dev = []
for i in sup_matrix:
    avg_param_comb_sup_dev.append(sum(i) / len(i))

# # avg, min, max, std
# sum(avg_param_comb_sup_dev) / len(avg_param_comb_sup_dev)
# min(avg_param_comb_sup_dev)
# max(avg_param_comb_sup_dev)
# np.std(avg_param_comb_sup_dev)

avg_param_comb_unsup_dev = []
for i in unsup_matrix:
    avg_param_comb_unsup_dev.append(sum(i) / len(i))

avg_param_comb_rf_dev = []
for i in unsup_matrix_forest:
    avg_param_comb_rf_dev.append(sum(i) / len(i))

# avg, min, max, std
sum(avg_param_comb_rf_dev) / len(avg_param_comb_rf_dev) # 0.8349
min(avg_param_comb_rf_dev) # 0.8255
max(avg_param_comb_rf_dev) # 0.8575
np.std(avg_param_comb_rf_dev) # 0.0121


str_ticks = [str(i[0]) + "," + str(i[1]) + "," + str(i[2]) for i in param_comb_list]
plt.plot(x_axis, avg_param_comb_sup_dev)
plt.plot(x_axis, avg_param_comb_unsup_dev)
plt.plot(x_axis, avg_param_comb_rf_dev)
plt.title("avg ndcg vs param combinations")
plt.xlabel("param combs: lr, wordngram, ws")
plt.ylabel("avg ndcg")
plt.legend(["sup", "unsup ridge", "unsup rf"])
plt.xticks(x_axis, str_ticks, rotation="vertical")
plt.grid()
plt.show()

# avg_param_comb_sup_dev[9]  # 0.6552 [0.05, 5, 5]
# avg_param_comb_sup_dev[10] # 0.7829 [0.05, 5, 8]
# avg_param_comb_sup_dev[19]  # 0.6552 [0.1, 5, 5]
# avg_param_comb_sup_dev[20] # 0.8581 [0.1, 5, 8]

## 2.number of labels
num_lab_dev = []
for i in labels_list:
    num_lab_dev.append(len(i))

# for table
table_data = [0 for i in range(13)]
for i in num_lab_dev:
    table_data[i-1] += 1

plt.hist(num_lab_dev)
plt.xticks(range(1, 13))
plt.title("distribution of number of labels")
plt.xlabel("number of labels per doc")
plt.ylabel("number of docs")
plt.show()

# how each param set perform on certain number of labels
target_matrix = unsup_matrix_forest
label_examine = [[[] for j in range(70)] for i in range(13)]
for i,j in enumerate(target_matrix): # i is 0 - 69, j is each 29912 vector
    for m, n in enumerate(j): # m is 0 - 29912, n is ndcg value
        label_examine[num_lab_dev[m] - 1][i].append(n)

label_examine_summary = []
for i in label_examine:
    label_examine_summary.append(map(np.average, i))

x_axis = [i for i in range(1,71)]
for i in label_examine_summary:
    plt.plot(x_axis, i)

# with open("label_examine_summary.json", "w") as f:
#     f.write(json.dumps(label_examine_summary))

plt.title("effect of number of labels in each param comb")
plt.xlabel("param combs: lr, wordngram, ws")
plt.ylabel("avg ndcg")
plt.legend([str(i) + " labels" for i in range(1,14)])
plt.xticks(x_axis, str_ticks, rotation="vertical")
plt.grid()
plt.show()

num_lab = {}
for i,j in enumerate(num_lab_dev):
    if j in num_lab:
        num_lab[j].append(tfidf_vec[i])
    else:
        num_lab[j] = [tfidf_vec[i]]

num_of_lab = [i for i in range(1,14)]
tfidf_avg_ndcg = [0 for i in range(1,14)]
num_count = [0 for i in range(1,14)]

for i,j in num_lab.items():
    num_count[i - 1] = len(j)
    tfidf_avg_ndcg[i - 1] = np.average(j)

x_tick = [str(i+1)+","+str(j) for i,j in enumerate(num_count)]

plt.bar(num_of_lab, tfidf_avg_ndcg)
plt.title("avg ndcg group by num of labels per doc - tfidf dev")
plt.xlabel("number of labels per doc")
plt.ylabel("avg ndcg")
plt.xticks(num_of_lab, x_tick, rotation="vertical")
plt.show()

# there are 3 categories of params. for each param, we fix a,b change c, see how different c influence the result (e.g. avg ndcg)
# multi and single theme comparison

## t-test
# result_matrix_correct, supervised exp, see if the average ndcg of other parameter combinations is indeed better than deafault params
from scipy import stats

# supervised
result_matrix_correct = json.loads(open("result_matrix_correct.json").read())

ttest_result = []
for i in range(len(result_matrix_correct)):
    ttest_result.append(list(stats.ttest_ind(result_matrix_correct[i], result_matrix_correct[0])))

# unupervised
unsup_matrix = json.loads(open("unsup_matrix.json").read())
ttest_result_unsup = []
for i in range(len(unsup_matrix)):
    ttest_result_unsup.append(list(stats.ttest_ind(unsup_matrix[i], unsup_matrix[0])))

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='zhtpandog', api_key='4WSyFuTXsmMkxlnqP0a5')

idx = [i for i in range(len(result_matrix_correct))]
t_value_sup = [i[0] for i in ttest_result]
p_value_sup = [i[1] for i in ttest_result]

t_value_unsup = [i[0] for i in ttest_result_unsup]
p_value_unsup = [i[1] for i in ttest_result_unsup]

t_value = t_value_sup
p_value = p_value_sup

trace1 = go.Scatter(
    x = idx,
    y = t_value,
    name = "t_value"
)

trace2 = go.Scatter(
    x = idx,
    y = p_value,
    name = "p_value",
    yaxis = "y2"
)

data = [trace1, trace2]

layout = go.Layout(
    title='Trend for t_value and p_value for t-test (supervised)',
    yaxis=dict(
        title='t_value'
    ),
    yaxis2=dict(
        title='p_value',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
)
fig3 = go.Figure(data=data, layout=layout)
plot_url3 = py.plot(fig3, filename='ttest_supervised')

# https://plot.ly/~zhtpandog/0/trend-for-t-value-and-p-value-for-t-test-unsupervised/
# https://plot.ly/~zhtpandog/2/trend-for-t-value-and-p-value-for-t-test-supervised/

## 1,2,3,... label result for each model
# pick out the best supervised and unsupervised model
# num_lab_dev
# avg_param_comb_sup_dev
# avg_param_comb_unsup_dev

best_ndcg_s, best_idx_s = 0, 0 # best_ndcg_s = 0.9062, best_idx_s = 60 ([1.0, 1, 5])
for i in range(len(avg_param_comb_sup_dev)):
    if avg_param_comb_sup_dev[i] > best_ndcg_s:
        best_ndcg_s = avg_param_comb_sup_dev[i]
        best_idx_s = i
best_all_s = sup_matrix[60]

best_ndcg_u, best_idx_u = 0, 0 # best_ndcg_u = 0.9008, best_idx_s = 5 ([0.05, 3, 8])
for i in range(len(avg_param_comb_unsup_dev)):
    if avg_param_comb_unsup_dev[i] > best_ndcg_u:
        best_ndcg_u = avg_param_comb_unsup_dev[i]
        best_idx_u = i
best_all_u = unsup_matrix[5]

best_ndcg_rf, best_idx_rf = 0, 0 # best_ndcg_u = 0.9493, best_idx_s = 5 ([0.05, 3, 8])   best_idx = 9 [0.05, 5, 8]
for i in range(len(avg_param_comb_rf_dev)):
    if avg_param_comb_rf_dev[i] > best_ndcg_rf:
        best_ndcg_rf = avg_param_comb_rf_dev[i]
        best_idx_rf = i
best_all_rd = unsup_matrix_forest[5]

# best_all_s, best_all_u, best_all_rd, tfidf_vec
# first partition doc indices into buckets of different number of labels
num_label_buckets = [[] for i in range(13)]
for i in range(len(num_lab_dev)):
    num_label_buckets[num_lab_dev[i] - 1].append(i)

# sup, unsup, tfidf
sup_data, unsup_data, rf_data, tfidf_data = [[] for i in range(13)], [[] for i in range(13)], [[] for i in range(13)], [[] for i in range(13)]
for i,j in enumerate(num_label_buckets): # i is 0,1,2,...,12, for each # labels, j is all indices for part of 29912 docs that fall into each # labels partition
    for k in j: # k is document index, each one of 29912
        sup_data[i].append(best_all_s[k])
        unsup_data[i].append(best_all_u[k])
        tfidf_data[i].append(tfidf_vec[k])
        rf_data[i].append(best_all_rd[k])

sup_data_plot = [sum(i) / len(i) for i in sup_data]
unsup_data_plot = [sum(i) / len(i) for i in unsup_data]
tfidf_data_plot = [sum(i) / len(i) for i in tfidf_data]
rf_data_plot = [sum(i) / len(i) for i in rf_data]

str_ticks2 = [i for i in range(1,14)]
x_axis2 = [i for i in range(1,14)]
plt.plot(x_axis2, sup_data_plot)
plt.plot(x_axis2, unsup_data_plot)
plt.plot(x_axis2, rf_data_plot)
plt.plot(x_axis2, tfidf_data_plot)
plt.title("4 model avg ndcg for each #labels")
plt.xlabel("#labels")
plt.ylabel("avg ndcg in the #label group")
plt.legend(["sup", "ridge", "rf", "tfidf"])
plt.xticks(x_axis2, str_ticks2)
plt.show()

# random forest comb #5: 0.8537 default param
ndcg_grid_matrix_1 = json.loads(open("ndcg_grid_matrix_1.json").read())

# find optimum param comb based on avg ndcg
from sklearn.model_selection import ParameterGrid
param_dict = {"n_estimators": [10, 20, 30, 40, 50], "criterion": ["gini", "entropy"],
                  "max_features": ["auto", "sqrt", "log2", None], "n_jobs": [-1]}
param_list = list(ParameterGrid(param_dict))

result_matrix = ndcg_grid_matrix_1

param_n_avg_ndcg = []
best_ndcg = 0 # 0.8740 -> 2% boost
best_param = None # {'max_features': 'auto', 'n_estimators': 50, 'n_jobs': -1, 'criterion': 'entropy'}
# default: {'avg_ndcg': 0.855967579881833, 'params': {'max_features': 'auto', 'n_estimators': 10, 'n_jobs': -1, 'criterion': 'gini'}}
for idx, res in enumerate(result_matrix):
    avg_ndcg = sum(res) / len(res)
    param_n_avg_ndcg.append({"params": param_list[idx], "avg_ndcg": avg_ndcg})
    if avg_ndcg > best_ndcg:
        best_ndcg = avg_ndcg
        best_param = param_list[idx]

param_list_l = []
for i in param_list:
    vals = i.values()
    vals_str = [str(j) for j in vals if vals != -1]
    param_list_l.append(",".join(vals_str))

ndcg_list = []
for i in param_n_avg_ndcg:
    ndcg_list.append(i["avg_ndcg"])

x_axis = [i for i in range(1,41)]

plt.plot(x_axis, ndcg_list)
plt.title("avg ndcg vs rf param combinations")
plt.xlabel("max_features, n_estimators, criterion")
plt.ylabel("avg ndcg")
# plt.legend(["sup", "unsup ridge", "unsup rf"])
plt.xticks(x_axis, param_list_l, rotation="vertical")
plt.grid()
plt.show()