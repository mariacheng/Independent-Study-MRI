# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:55:07 2019

@author: Maria

Load feature data and train models
"""
from data.utils import BASE_DIR
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler


def model_and_predict(model, x_train, y_train, x_test, y_test, standardize=True, normalize=False, doSample=True, sample='under'):
    rus = RandomUnderSampler()
    if doSample:
        print('Sampling')
        x_train, y_train = rus.fit_sample(x_train, y_train)

    if standardize:
        print('Standardizing...')
        sc = StandardScaler().fit(x_train)
        x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)
    if normalize:
        print('Normalizing...')
        x_train = MinMaxScaler().fit_transform(x_train)
        x_test = MinMaxScaler().fit_transform(x_test)

    # Train model

    print('Training on model...')
    mod = models[model]
    mod.fit(x_train, y_train)

    print('Evaluating on model...')
    #y_pred = mod.predict(x_test)
    y_pred = mod.predict_proba(x_test)[:,1]

    fpr, tpr, thresh = roc_curve(y_test, y_pred)
    auc_val = auc(fpr, tpr)
    print(auc_val)

    i = np.argmax((1. - fpr) + tpr)
    threshold = thresh[i]
    if model == 'svm':
        y_pred_bin = mod.predict(x_test)
    else:
        y_pred_bin = y_pred > threshold
#    conf_mat = confusion_matrix(y_test, y_pred_bin)
#    sens = conf_mat[0,0]/(conf_mat[1,0] + conf_mat[0,0])
#    spec = conf_mat[1,1]/(conf_mat[0,1] + conf_mat[1,1])
    tp = (y_test & y_pred_bin).sum()
    tn = ((~y_test) & (~y_pred_bin)).sum()
    fp = ((~y_test) & y_pred_bin).sum()
    fn = (y_test & (~y_pred_bin)).sum()
    sens = tp/(fn + tp)
    spec = tn/(fp + tn)

    return sens, spec, auc_val

# 5 folds

model_modalities = ['HBV']
models = {
	"knn": KNeighborsClassifier(n_neighbors=1),
	"naive_bayes": GaussianNB(),
	"logit": LogisticRegression(solver="lbfgs"),
	"svm": SVC(kernel="linear"),
	"decision_tree": DecisionTreeClassifier(),
	"random_forest": RandomForestClassifier(n_estimators=100),
	"mlp": MLPClassifier()
}
model = 'knn'
wZone = 'wZone'
file_savetype = 'slice'
aucs = []
sens = []
specs = []

for i in range(5):
    fold_files = BASE_DIR.glob('./fold*.npy')
    test_id = i+1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for file in fold_files:
        filecomp = file.stem.split('_')
        fold = int(filecomp[1])
        modality = filecomp[2]
        zone_lab = filecomp[-3]
        savetype = filecomp[-2]
        if savetype == file_savetype and wZone == zone_lab:
            data = np.load(file)
            if modality in model_modalities:
                if fold == test_id:
                    if filecomp[-1] == 'features':
                        x_test.extend(data)
                    else:
                        y_test.extend(data)
                else:
                    if filecomp[-1] == 'features':
                        x_train.extend(data)
                    else:
                        y_train.extend(data)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    sen,spec,auc1 = model_and_predict(model, x_train, y_train, x_test, y_test)
    aucs.append(auc1)
    sens.append(sen)
    specs.append(spec)

mean_auc = sum(aucs) / float(len(aucs))
mean_sens = sum(sens) / float(len(sens))
mean_spec = sum(specs) / float(len(specs))
print(mean_sens, mean_spec, mean_auc)
