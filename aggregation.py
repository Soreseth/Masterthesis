import torch
import numpy 
from sklearn import svm
from xgboost import XGBClassifier

def xgboost(X_train,y_train,X_test,y_test):
    
    clf = XGBClassifier(tree_method="hist", early_stopping_rounds=5)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    return clf

def svm_classifier(X_train, y_train):
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    return clf