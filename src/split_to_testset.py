#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:26:40 2019

@author: ruoqi
"""
import numpy as np 
import pandas as pd 
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
    path = "/datasets/home/96/996/ruz005/"
    train_df = pd.read_csv(path + "test.csv")
    X = train_df['Image']
    y = train_df['Id']
    for i in X:
        mymovefile(path+"whaletrain/"+i, path+"whaletest/"+i)
