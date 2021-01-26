# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 23:51:26 2021

@author: Anand
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
from sklearn.ensemble import BaggingClassifier

##################### DATA READ AND CLEAN ##################################

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
    datx=bcx1[3000:600000]
    daty=bcy1[3000:600000]
    datz=bcz1[3000:600000]
    
    # range is decided on 42000-3000 / 500, 500 hz is the sensor sampling rate of the ADC
    
    x_a=np.split(datx,1194)
    y_a=np.split(daty,1194)
    z_a=np.split(datz,1194)
    
    # w = blackman(len(datx))
    x = abs(fft(x_a))
    y= abs(fft(y_a))
    z = abs(fft(z_a))
    
    return x,y,z

###################### SUPER LEARNER MAKE #######################################


# create a list of base-models
def get_models():
    models = list()
    models.append(DecisionTreeClassifier())
    models.append(GaussianNB())
    models.append(KNeighborsClassifier(n_neighbors=4))
    models.append(AdaBoostClassifier())
    models.append(svm.NuSVC(gamma='auto'))
    models.append(RandomForestClassifier(n_estimators=1000))
    models.append(ExtraTreesClassifier(n_estimators=5000))
    # models.append(BaggingClassifier(n_estimators=10))
    # models.append(LogisticRegression(solver='liblinear'))    
    return models

# create the super learner
def get_super_learner(X):
	ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=True, sample_size=len(X))
	# add base models
	models = get_models()
	ensemble.add(models)
	# add the meta model
# 	ensemble.add_meta(svm.NuSVC(gamma='auto'))
	ensemble.add_meta(LogisticRegression(solver='liblinear'))
	return ensemble


##################### DATA PREPARE AND FEATURE PREPARE ########################

x1,x2,x3=data_read("baseline.txt")
y1,y2,y3=data_read("browser.txt")
v1,v2,v3=data_read("DeepLearning1.txt")



dim=len(x1)


c0_label=np.array(np.zeros(dim)).reshape((dim,1))
c1_label=np.array(np.ones(dim)).reshape((dim,1))
c2_label=np.array(np.ones(dim)*2).reshape((dim,1))
lab= np.concatenate((c0_label,c1_label,c2_label))


dat_X= np.concatenate((x1,y1,v1))
dat_Y= np.concatenate((x2,y2,v2))
dat_Z= np.concatenate((x3,y3,v3))

X_scaled=preprocessing.normalize(dat_X+dat_Y+dat_Z, norm='l2', axis=1)

train_data = []
for i in range(len(X_scaled)):
    # hist_arr, bin_edges = np.histogram(X_scaled[i], bins=10, density=True, range=(0,0.01))
    hist_arr, bin_edges = np.histogram(X_scaled[i], bins=10, density=True)
    
    # print(hist_arr)
    train_data.append(hist_arr)

train_data = np.array(train_data)


X1, X1_val, y1, y1_val = train_test_split(train_data, lab, test_size=0.10)


################### ENSEMBLE GENERATE #######################################

ensemble1 = get_super_learner(X1)

# fit the super learner
ensemble1.fit(X1, y1.ravel())
filename = 'tremors_model.sav'
pickle.dump(ensemble1, open(filename, 'wb'))

# summarize base learners
print(ensemble1.data)


# make predictions on hold out set
yhat1 = ensemble1.predict(X1_val)
print('Super Learner: %.3f' % (accuracy_score(y1_val, yhat1) * 100))

