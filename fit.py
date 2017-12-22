# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 14:00:47 2017

@author: BHRS-ZY-PC
"""
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.optimize import curve_fit 

ratio = tick / S0
x = ratio
y = volDifflist

def func(x, a, b, c):  
    return a*np.sin(x+b)+c
def func1(x, a, b):  
    return a*x + b
popt, pcov = curve_fit(func1, x, y)  
a = popt[0]   
b = popt[1]  
#c = popt[2]

yvals = func1(x,a,b)
plot1 = plt.plot(x, y, 's',label='original values')  
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')  
plt.xlabel('x')  
plt.ylabel('y')  
plt.legend(loc=4) #指定legend的位置右下角  
plt.title('curve_fit')  
plt.savefig('test3.png')  
plt.show()  