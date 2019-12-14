import numpy as np
import os
import sklearn.linear_model
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib
matplotlib.use('Agg')
import sklearn.svm
from sklearn.metrics import mean_squared_error


# read in the excel
df = pd.read_csv("./videos.csv")
sr = list(df["Sickness Rating"])
title = list(df["Title"])


length = 100000
name_list = []
max_val, min_val = -1, 1000000
for root, dirs, files in os.walk("/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/"):
    for file in files:
        if not file.endswith(".npy"):
            continue
        path = os.path.join(root, file)
        arr = np.load(path)
        if len(arr) > 2000:
            continue
        name_list.append(path)

        length = min(length, len(arr))
        max_val = max(max_val, np.amax(arr))
        min_val = min(min_val, np.amin(arr))

print(max_val,min_val)
# prepare data for future use
x, y = [], []
for name in name_list:
    # clip size
    data = np.load(name)[:length]

    # normalize
#    data = (data - np.amin(data)) / (np.amax(data) - np.amin(data))
    data = (data - min_val) / (max_val - min_val)
    print(np.amin(data), np.amax(data))
    x.append(data)
    i = title.index(name.split("_")[-1][:-4])
    y.append(float(sr[i]))

# cross validation
avg_score, rmse = 0, 0
time = 50
for _ in range(time):
    index = list(range(len(x)))
    random.shuffle(index)
    xtrain, xtest, ytrain, ytest = [], [], [], []
    for i in index:
        if len(xtest) < 4:
            xtest.append(x[i])
            ytest.append(y[i])
        else:
            xtrain.append(x[i])
            ytrain.append(y[i])

    model = sklearn.svm.SVR(gamma="scale", C=0.1).fit(xtrain,ytrain)
    avg_score += model.score(xtest, ytest) / time
    rmse += mean_squared_error(ytest, model.predict(xtest)) ** 0.5 / time
    #print(model.predict(xtrain))
#    print(model.score(xtest, ytest), mean_squared_error(ytest, model.predict(xtest)) ** 0.5 )

    # test variable importance
#     # use test image 1
#     _data, _label = xtest[0].reshape(1,-1), ytest[0]
#     print(index[0])
#     base_score = model.predict(_data)
#     print("Before changing variable, the sickness score prediction is ", base_score, " and the label is ", _label)
#     difference = []
#     for i, number in enumerate(_data[0]):
# #        print("changing number ", i-20, " to ", i+20, " to zero")
#         temp = np.array([_data[0][t] if i-20<=t<=i+20 else 0 for t in range(len(_data[0]))]).reshape(1,-1)
#         difference.append(abs(model.predict(temp)-base_score))
#     plt.plot(difference)
#     plt.savefig("differences")
#     print("After changing variable, the sickness score prediction is ", base_score, " and the label is ", _label)

print(avg_score, rmse)