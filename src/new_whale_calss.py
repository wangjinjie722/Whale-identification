#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:13:49 2019

@author: ruoqi
"""
import numpy as np
import pandas as pd

if __name__ == "__main__":
    path = "../"
    train_df = pd.read_csv(path + "train.csv")
    test_df = pd.read_csv(path + "test.csv")
    y_tr = train_df['Id']
    y_ts = test_df['Id']
    X_ts = test_df['Image']
    num_tr = len(set(y_tr))
    num_ts = len(set(y_ts))
    print("train_set classes: ", num_tr)
    print("test_set classes: ", num_ts)
    print("test_set classes not in train_set: ", num_ts - len(set(y_tr)&set(y_ts)))
    for i in range(len(y_ts)):
        if y_ts[i] not in set(y_tr):
            y_ts[i] = 'new_whale'
    y_test = pd.DataFrame(data=y_ts.values, columns=['Id'])
    X_test = pd.DataFrame(data=X_ts.values, columns=['Image'])
    testfile = pd.merge(X_test, y_test, left_index=True, right_index=True)
    testfile.to_csv(path + "new_whale_test.csv",index=False)