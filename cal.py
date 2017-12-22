# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 17:37:26 2017

@author: BHRS-ZY-PC
"""
from __future__ import division
import pandas as pd
import copy
import statsmodels.api as sm
import numpy as np

ratio = np.array(tick) / np.array(S0)
ratioTest = ratio.tolist()
QTest1 = Q.tolist()
QTest2 = copy.deepcopy(QTest1)  
 
len1 = len(ratioTest)    
len2 = len(sigma)
    
for j in range(len2-1):
    ratioTest.extend(ratio)
    QTest2.extend(QTest1)

sigmaTest = []
for i in range(len2):
    for j in range(len1):
#        sigmaTest.append(-0.2227*np.sin(sigma[i]+1.176)+0.2435)
        sigmaTest.append(sigma[i])
X = pd.DataFrame([ratioTest, sigmaTest]).T
#X = pd.DataFrame([ratioTest, sigmaTest]).T
#X.iloc[:,0] = (X.iloc[:,0]) **1.05 * 178.4938
#X.iloc[:,2] = X.iloc[:,2]**0.26*0.0197
volDiffTest=[]
for i in range(len(volDifflist)):
    for j in range(len(volDifflist[i])):
        volDiffTest.append(volDifflist[i][j])
Y = pd.DataFrame(volDiffTest)

X = sm.add_constant(X)
est=sm.OLS(Y,X).fit()