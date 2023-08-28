import csv

import sys
import os
import pickle
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, precision_recall_fscore_support

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import DataStructs
import base64
import fcntl
import rdkit.Chem.MolStandardize as rdMolStandardize

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def convert_to_indices(num, shape):
    indices = []
    for dim in reversed(shape):
        indices.append( int(num % dim) )
        num //= dim
    indices.reverse()
    return indices

def readcsv(fname, keyname):
    df=pd.read_csv(fname, sep='\t')
    data=[]
    fields=keyname.split('_')
    bitsize=int(fields[2])

    for m in df[keyname]:
        try:
            bv=DataStructs.cDataStructs.ExplicitBitVect(bitsize)
            bv.FromBase64(m)
            data.append(np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(bv), dtype=np.uint8), bitorder='little'))
        except:
            continue
    return(data)



r=3
b=2048

keyname='MorganFP_' + str(r) + '_' + str(b)

covalent_training_set=readcsv('morgan_data/trainingset_covalent_morgan.csv', keyname)
noncovalent_training_set=readcsv('morgan_data/trainingset_noncovalent_morgan.csv', keyname)

X_data=covalent_training_set + noncovalent_training_set
y_data=[1]*len(covalent_training_set) + [0]*len(noncovalent_training_set)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=1)

#clf=HistGradientBoostingClassifier(max_leaf_nodes=41, max_depth=None, class_weight="balanced")
#clf=LogisticRegression(class_weight="balanced", solver='saga')
#clf=RandomForestClassifier(max_depth=21)
clf=SVC(kernel="rbf", C=1)

testset_covalent=readcsv('morgan_data/testset_external_positive_morgan.csv', keyname)
testset_noncovalent=readcsv('morgan_data/testset_external_negative_morgan.csv', keyname)

X_test_ext=testset_covalent + testset_noncovalent
y_test_ext=[1]*len(testset_covalent) + [0]*len(testset_noncovalent)

clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

print('Training set AUC : ' + str(roc_auc_score(y_train, y_train_pred)))

y_test_pred = clf.predict(X_test)
auc_testset=roc_auc_score(y_test_pred, y_test)
print('AUC test (internal): ' + str(auc_testset))
metrics_test_internal=precision_recall_fscore_support(y_test, y_test_pred, average='binary')
print('Precision test (internal) : ' + str(metrics_test_internal[0]))
print('Recall test (internal) : ' + str(metrics_test_internal[1]))
print('F1 test (internal) : ' + str(metrics_test_internal[2]))
print(y_test_pred)
print('internal confusion matrix')
cm = confusion_matrix(y_test_pred, y_test)
print(cm)
y_pred_ext = clf.predict(X_test_ext)
cm = confusion_matrix(y_test_ext, y_pred_ext)

auc=roc_auc_score(y_pred_ext, y_test_ext)

print('AUC external: ' + str(auc))
print(precision_recall_fscore_support(y_test_ext, y_pred_ext, average='binary'))
metrics_test_external=precision_recall_fscore_support(y_test_ext, y_pred_ext, average='binary')
print('Precision test (external) : ' + str(metrics_test_external[0]))
print('Recall test (external) : ' + str(metrics_test_external[1]))
print('F1 test (external) : ' + str(metrics_test_external[2]))
print('external_confusion_matrix')
print(cm)

testset=readcsv('morgan_data/testset-negative_false_covalent.csv', keyname)
y_testset=clf.predict(testset)
false_positive_false_covalent=sum(y_testset)

print('decoy : ' + str(false_positive_false_covalent))
print('decoy rate : ' + str(float(false_positive_false_covalent) / len(y_testset)))
