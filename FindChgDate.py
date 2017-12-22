# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:33:11 2017

@author: BHRS-ZY-PC
"""
from WindPy import w
import pandas as pd
import datetime
import numpy as np


def APIDataToPandas(x):
    
    fm=pd.DataFrame(x.Data[0],index=x.Times,columns=x.Fields)
    return fm
 
def OriginalLogRet(code, BeginDay, EndDay):
    ClosePriceSet= APIDataToPandas(w.wsd(code, 'close', BeginDay, EndDay))
    logReturn = ClosePriceSet.apply(np.log).diff(1).dropna()
   
    return logReturn

def OriginalLogRetOneDate(code, EndDay, D):
# 截止日往前推任意天的数据
    ClosePriceSet = APIDataToPandas(w.wsd(code, "close", "ED-{:d}TD".format(D), EndDay, ""))
    logReturn = ClosePriceSet.apply(np.log).diff(1).dropna()  
    
    return logReturn

    
def RevisedData(code,BeginDay,EndDay):
    OriData = APIDataToPandas(w.wsd(code, "trade_hiscode",BeginDay-datetime.timedelta(days=1),\
                                 EndDay,"Days=Tradingdays"))
    
    OriData = OriData.drop_duplicates()
    RevisedData = OriData.drop([OriData.index[0]])        
        
    return RevisedData

def RevisedLogRet(code,BeginDay,EndDay):
    RevisedLogReturn = OriginalLogRet(code,BeginDay,EndDay)
    RevData = RevisedData(code,BeginDay,EndDay)   
    l = len(RevData)
    
    for i in range(l):
        BeforeAndAfterChgPrices = w.wsd(RevData.iloc[i][0],"close", "ED-1TD",RevData.index[i]).Data[0]
        
        logReturn = np.log(BeforeAndAfterChgPrices[1]/BeforeAndAfterChgPrices[0])
        RevisedLogReturn.loc[RevData.index[i]] = logReturn
    return RevisedLogReturn
            
def volitility(code,T,numOfYearBefore):
    EndDay = datetime.date.today()
    BeginDay = EndDay - relativedelta(years=numOfYearBefore)
    LogRet = RevisedLogRet(code,BeginDay,EndDay)
    days = w.tdayscount(EndDay-relativedelta(months=T),EndDay, "").Data[0][0]
    vol = (LogRet.rolling(window=days,center=False).std() * np.sqrt(252)).dropna().iloc[-1][0]
    
    return vol
    
if __name__ == '__main__':
    w.start()
    code = "CU.SHF"
    start_date = datetime.date(2017,5,8)
    end_date = datetime.date(2017,12,12)
    a = RevisedLogRet(code,start_date,end_date)
    b = OriginalLogRet(code,start_date,end_date)