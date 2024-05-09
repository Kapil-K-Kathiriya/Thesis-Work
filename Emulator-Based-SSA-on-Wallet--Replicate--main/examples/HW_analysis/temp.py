#!/usr/bin/env python3
from rainbow.devices.stm32 import rainbow_stm32f215 as rainbow_stm32
import numpy as np
import pandas as pd
import random
import pickle
import os
from rainbow import TraceConfig, HammingWeight
from lascar import TraceBatchContainer, Session, NicvEngine
from rainbow.utils.plot import viewer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC


X = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [1, 2, 3, 9]]
X = np.array(X)
index = [] # it keeps track of the column indices that have at least one non-matching element with first row's element
count = [0,0,0,0]
mean = []
std = []
# print(X.shape[1])
for i in range(X.shape[1]):
    mean.append(np.mean(X[:,i]))
    std.append(np.std(X[:,i]))
    
mean = np.array(mean) # contains means of each column(total column=46)
std = np.array(std)
for i in range(X.shape[1]): #X.shape[1] is total column in X, here which is 46 
    for j in range(X.shape[0]): #X.shape[0] is total row in X, here which is 1500 
        if(mean[i]!= X[0,i]):
            count[i] = count[i]+1
    if(count[i]!=0):
        index.append(i)

print(count)

print(9-1)
print(1-9)