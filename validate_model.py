# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:21:52 2021

@author: Pallav
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import blackman
from scipy.fft import fft
from sklearn import preprocessing
import sklearn.preprocessing as pp
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlens.ensemble import SuperLearner
from statistics import median, mode, mean
import pandas as pd
import pickle
import math
import glob

def data_read(dataset):
    bcx1=[]
    bcy1=[]
    bcz1=[]
    g=open(dataset,"r")
    b=g.readlines()
    filtered = filter(lambda x: not re.match(r'^\s*$', x), b)
    for lines in filtered:
            # bx2,by2,bz2=msg_fmt_mis_vib(lines)
            polData=lines.strip()
            data1=re.sub('\s+',' ',polData)
            s1=data1.split(",")
            bx2=s1[1]
            by2=s1[2]
            bz2=s1[3]
            bcx1.append(bx2)
            bcy1.append(by2)
            bcz1.append(bz2)
    g.close()
    
    bcx1=np.array(bcx1)
    bcx1=bcx1.astype('float64')
    bcy1=np.array(bcy1)
    bcy1=bcy1.astype('float64')
    bcz1=np.array(bcz1)
    bcz1=bcz1.astype('float64')
    
    print(len(bcx1))
    print(math.floor(len(bcx1)/500)*500)
    datx=bcx1[3000:math.floor(len(bcx1)/500)*500]
    daty=bcy1[3000:math.floor(len(bcy1)/500)*500]
    datz=bcz1[3000:math.floor(len(bcz1)/500)*500]
    print(len(datx)/500)
    # range is decided on 42000-3000 / 500, 500 hz is the sensor sampling rate of the ADC
    
    x_a=np.split(datx,len(datx)/500)
    y_a=np.split(daty,len(daty)/500)
    z_a=np.split(datz,len(datz)/500)
    
    # w = blackman(len(datx))
    x = abs(fft(x_a))
    y= abs(fft(y_a))
    z = abs(fft(z_a))
    
    return x,y,z
############### FIT ON UNSEEN DATA ######################


with open('tremors_model.sav', 'rb') as f:
    ensemble1 = pickle.load(f)

res_file = open('result.txt','w')

files = glob.glob('unseen_data/*.txt')
for ff in files:
    u1,u2,u3=data_read(ff)
    # u1,u2,u3=data_read("uncontrolled_data/dl1.txt")
    
    u1=preprocessing.normalize(u1+u2+u3, norm='l2', axis=1)
    test_data = []
    for i in range(len(u1)):
      hist_arr, bin_edges = np.histogram(u1[i], bins=10, density=True)
      test_data.append(hist_arr)

    test_data = np.array(test_data)
    # evaluate meta model
    yhatA = ensemble1.predict(test_data)

    
    print(yhatA)
    print(np.floor(mean(yhatA)))
    print("Class is: "+str(mean(yhatA)))
    res_file.write(ff+'\n')
    res_file.write(str(yhatA)+'\n')
    res_file.write(str(np.floor(mean(yhatA)))+'\n')
    res_file.write("Class is: "+str(mean(yhatA))+'\n')

res_file.close()