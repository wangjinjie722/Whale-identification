#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:27:45 2019

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
    for i in range(len(y_ts)):
        if y_ts[i] not in set(y_tr):
            y_ts[i] = 'new_whale'
    y_test = pd.DataFrame(data=y_ts.values, columns=['Id'])
    X_test = pd.DataFrame(data=X_ts.values, columns=['Image'])
    testfile = pd.merge(X_test, y_test, left_index=True, right_index=True)
    testfile.to_csv(path + "new_whale_test.csv",index=False)