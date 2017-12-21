# -*- coding: utf-8 -*-                                                                                                     # -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:40:04 2017

@author: BHRS-ZY-PC
"""
from __future__ import division
from WindPy import w
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import * 
import FindChgDate


mpl.rcParams['font.sans-serif'] = ['SimHei']  


def APIDataToPandas(x):
    
    fm=pd.DataFrame(x.Data[0],index=x.Times,columns=x.Fields)
    return fm

class VolatilityCone(object):
    
    def __init__(self, BeginDay, EndDay):
        w.start()
        self.M = np.arange(30,270,30)
        self.BM = np.arange(22,198,22)
        self.BeginDay = BeginDay
        self.EndDay = EndDay

#    def OriginalReturnData(self, code):
#        ClosePriceSet= APIDataToPandas(w.wsd(code, 'close', self.BeginDay, self.EndDay))
#        logReturn = ClosePriceSet.apply(np.log).diff(1).dropna()
#    
#        return logReturn
    
    def Quantile(self, x):
        
        maxvolList = []
        minvolList = []
        ntpervolList = []       #90%
        sfpervolList = []       #75%
        medianvolList = []      #50%
        tfpervolList = []       #25%
        tpervolList = []        #10%
        
        l = len(self.BM)
        
        for i in range(l):
            
            vol = (x.rolling(window=self.BM[i],center=False).std() * np.sqrt(252)).dropna()
            maxvol    = np.max(vol)
            minvol    = np.min(vol)
            ntpervol  = np.percentile(vol,90)
            sfpervol  = np.percentile(vol,75)
            medianvol = np.percentile(vol,50)
            tfpervol  = np.percentile(vol,25)
            tpervol   = np.percentile(vol,10)
            
            maxvolList.append(maxvol)
            ntpervolList.append(ntpervol)
            sfpervolList.append(sfpervol)
            medianvolList.append(medianvol)
            tfpervolList.append(tfpervol)
            tpervolList.append(tpervol)
            minvolList.append(minvol)
            
        return maxvolList,ntpervolList,sfpervolList,medianvolList,tfpervolList,tpervolList,minvolList
    
    def plot(self,params,title):
        
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_xticks(self.M)
        ax.set_title(title)
        
        plt.plot(self.M, params[0], label=u'最大值')
        plt.plot(self.M, params[1], label=u'90%分位数')
        plt.plot(self.M, params[2], label=u'75%分位数')
        plt.plot(self.M, params[3], label=u'50%分位数')
        plt.plot(self.M, params[4], label=u'25%分位数')
        plt.plot(self.M, params[5], label=u'10%分位数')
        plt.plot(self.M, params[6], label=u'最小值')
        plt.legend(loc='upper right')
        plt.xlabel(u'剩余到期时间（天）')
        plt.ylabel(u'波动率')
        ax.grid(True,axis ='y', linewidth=0.5)
        plt.show()     
            
def main():
    
    
    EndDay = datetime.date.today()- datetime.timedelta(days=30) 
    BeginDay = EndDay - datetime.timedelta(days=365) * 2  
    
    VC = VolatilityCone(BeginDay, EndDay)
#    species = [
#                ['CU.SHF','AL.SHF','ZN.SHF'],
#                    
#                ['NI.SHF','I.DCE','J.DCE','A.DCE','C.DCE',
#                 'M.DCE','P.DCE','JD.DCE','L.DCE','PP.DCE'],
##********************郑商所所有合约从1609往后变更为609******************
#                ['RM.CZC','OI.CZC','SR.CZC','CF.CZC','TA.CZC'],
##                 'ZC.CZC'],
#                ['AU.SHF','AG.SHF'],
#                ['RB.SHF']
#              ]

    species = [
                'CU.SHF','AL.SHF','ZN.SHF','NI.SHF','I.DCE','J.DCE','A.DCE',
                'C.DCE','M.DCE','P.DCE','JD.DCE','L.DCE','PP.DCE','RM.CZC',
                'OI.CZC','SR.CZC','CF.CZC','TA.CZC','ZC.CZC','AU.SHF','AG.SHF',
                'RB.SHF'
              ]
    
        

    l1 = len(species)
    
    for i in range(l1):
        
        code = species[i] 
    
        RevisedLogReturn = FindChgDate.RevisedLogRet(code, BeginDay, EndDay)
        
        param = VC.Quantile(RevisedLogReturn)
        VC.plot(param, code)
    
if __name__ == '__main__':
    main()  
    
    
    
    
    
#*********************************************************************************************************************    
#    
#    
#    
#def volatilityConeM(code, freq=1):      #分钟级别数据
#    w.start()
#   
#    M = np.arange(30,270,30)
#    multiplier = len(w.wsi(code, 'close', '2017-11-23 0:00:00', '2017-11-23 23:59:59', BarSize=freq).Data[0])
#    EndTime = datetime.datetime.combine(datetime.date.today()-datetime.timedelta(days=1), datetime.time.max)
#    BeginTime = datetime.datetime.combine(EndTime - datetime.timedelta(days=365) * 2, datetime.time.min)
#    closepricesetlist = w.wsi(code, 'close', BeginTime, EndTime, BarSize=freq)
#    logReturn = pd.Series(closepricesetlist.Data[0]).apply(np.log).diff(1)
#    
#    maxvolList = []
#    minvolList = []
#    ntpervolList = []       #90%
#    sfpervolList = []       #75%
#    medianvolList = []      #50%
#    tfpervolList = []       #25%
#    tpervolList = []        #10%
#    
#    for i in range(len(M)):
#        windowNum = M[i]*multiplier
#        
#        #有些时间点不存在价格，会导致在rolling计算的时候使结果为NAN，故这里使用min_periods
#        #来进行约束，从而不会使vol整体为NAN，这里取13/16只是近似约束，后续可继续讨论
#        vol = (logReturn.rolling(window=windowNum,min_periods=int(np.ceil(windowNum*13/16)),\
#                                 center=False).std() * np.sqrt(252*multiplier)).dropna()
#        maxvol    = np.max(vol)
#        minvol    = np.min(vol)
#        ntpervol  = np.percentile(vol,90)
#        sfpervol  = np.percentile(vol,75)
#        medianvol = np.percentile(vol,50)
#        tfpervol  = np.percentile(vol,25)
#        tpervol   = np.percentile(vol,10)
#        
#        maxvolList.append(maxvol)
#        ntpervolList.append(ntpervol)
#        sfpervolList.append(sfpervol)
#        medianvolList.append(medianvol)
#        tfpervolList.append(tfpervol)
#        tpervolList.append(tpervol)
#        minvolList.append(minvol)
#        
#    w.stop()
#    return maxvolList,ntpervolList,sfpervolList,medianvolList,tfpervolList,tpervolList,minvolList
#
#    

    
    
    
    
    
    