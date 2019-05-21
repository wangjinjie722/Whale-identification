#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:26:41 2019

@author: ruoqi
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split

import os,shutil

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print("copy %s -> %s"%( srcfile,dstfile))


if __name__ == "__main__":
    path = "/Users/ruoqi/Downloads/whale/input/"
    train_df = pd.read_csv(path + "train.csv")
    X = train_df['Image']
    y = train_df['Id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    for i in X_test:
        mymovefile(path+"train/"+i, path+"test/"+i)
    y_ts = pd.DataFrame(data=y_test.values, columns=['Id'])
    X_ts = pd.DataFrame(data=X_test.values, columns=['Image'])
    testfile = pd.merge(X_ts, y_ts, left_index=True, right_index=True)
    y_tr = pd.DataFrame(data=y_train.values, columns=['Id'])
    X_tr = pd.DataFrame(data=X_train.values, columns=['Image'])
    trainfile = pd.merge(X_ts, y_ts, left_index=True, right_index=True)
    testfile.to_csv(path + "test.csv",index=False)
    trainfile.to_csv(path + "train.csv",index=False)   