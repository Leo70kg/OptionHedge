
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:17:29 2017

@author: BHRS-ZY-PC
"""

from __future__ import division
import numpy as np
from numpy import random
import pandas as pd
import xlwings as xw
from scipy.stats import norm
import datetime
from WindPy import w
import re
import json
import FindChgDate
from dateutil.relativedelta import *

def numRoundArr(num, tick):
#根据不同的tick调整价格 , num格式为narray
    row = num.shape[0]
    col = num.shape[1]
    roundNum = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if num[i, j] % tick > tick/2:
                roundNum[i,j] = num[i,j] - num[i,j] % tick + tick
            else:
                roundNum[i,j] = num[i,j] - num[i,j] % tick
        
    return roundNum

def numRound(num, tick):
    if num % tick > tick/2:
        roundNum = num - num % tick + tick
    else:
        roundNum = num - num % tick
        
    return roundNum

class MCpricing(object):
        
    def MCsimulation(self,S0,T,sigma,M,N,mu=0):
    #股票实际价格模拟
        random.seed(seed=123)
        X = random.randn(M, N)
        deltaT = T/252 / N
        e = np.exp((mu-0.5*sigma**2) * deltaT  + sigma * np.sqrt(deltaT) * X)
        ST = np.cumprod(np.c_[S0 * np.ones((M,1)), e], axis=1) 
    
        return ST    
        
    def priceOptionMC(self,S0,K,r,T,sigma,M,N,payoff_function):
        """
        priceOptionMC: Black-Scholes price of a generic option providing a payoff.
        INPUT:
            S0 : Initial value of the underlying asset
            r : Risk-free interest rate 
            T : Time to expiry 
        sigma : Volatility 
            M : Number of simulations
            N : Number of observations
        payoff_function : payoff function of the option
    
        OUTPUT:
            price_MC : MC estimate of the price of the option in the Black-Scholes model  
     
        """
        ## Generate M x N samples from N(0,1)
        random.seed(seed=123)
        X = np.random.randn(M, N)

        ## Simulate M trajectories in N steps
        deltaT = T/252 / N
        e = np.exp((r-0.5*sigma**2) * deltaT  + sigma * np.sqrt(deltaT) * X)
        S = np.cumprod(np.c_[S0 * np.ones((M,1)), e], axis=1)        
        
        ## Compute the payoff for each trajectory
        payoff = payoff_function(S)

        ## MC estimate of the price and the error of the option
        discountFactor = np.exp(-r*T/252)
    
    
        price_MC = discountFactor * np.mean(payoff)
        
        return price_MC
    
        
    def priceBarrierUpAndOutCallMC(self,S0,K,B,r,T,sigma,M,N):
        def payoff(S):
            return np.where(S[:, -1] < K, 0, S[:, -1] - K) * np.all(S < B, axis=1)
        return self.priceOptionMC(S0,K,r,T,sigma,M,N,payoff)
    
    def priceBarrierUpAndOutPutMC(self,S0,K,B,r,T,sigma,M,N):
        def payoff(S):
            return np.where(S[:, -1] > K, 0, K - S[:, -1]) * np.all(S < B, axis=1)
        return self.priceOptionMC(S0,K,r,T,sigma,M,N,payoff)

    def priceAsianArithmeticMeanCallMC(self,S0,K,r,T,sigma,M,N):
        
        def payoff(S):
            S_ar_mean = np.mean(S[:, 1:], 1)
            return np.where(S_ar_mean < K, 0, S_ar_mean - K)
        return self.priceOptionMC(S0,K,r,T,sigma,M,N,payoff)
    
    def priceAsianArithmeticMeanPutMC(self,S0,K,r,T,sigma,M,N):
        
        def payoff(S):
            S_ar_mean = np.mean(S[:, 1:], 1)
            return np.where(S_ar_mean > K, 0, K - S_ar_mean)
        return self.priceOptionMC(S0,K,r,T,sigma,M,N,payoff)
        
    def priceEuropeanCallMC(self,S0,K,r,T,sigma,M,N):
    
        def payoff(S):
            return np.where(S[:,-1] < K, 0, S[:,-1] - K)
        return self.priceOptionMC(S0,K,r,T,sigma,M,N,payoff)
    
    def priceEuropeanPutMC(self,S0,K,r,T,sigma,M,N):
    
        def payoff(S):
            return np.where(S[:,-1] > K, 0, K - S[:,-1])
        return self.priceOptionMC(S0,K,r,T,sigma,M,N,payoff)
    
class Greeks(object):
    
    def numericalDerivative(self, f, x0, h):
        
        if x0 == 0:
            derivative = (f(h) - f(-h))/(2 * h)
        else:
            derivative = (f(x0 * (1+h)) - f(x0 * (1-h)))/(2 * x0 * h)
        return derivative
       
    def delta(self,S0,K,r,T,sigma,M,N):
        
        pricingEngine = MCpricing()
        h = 1e-5
        def fPrice(S):
            return pricingEngine.priceEuropeanCallMC(S,K,r,T,sigma,M,N)
        
        delta = self.numericalDerivative(fPrice, S0, h)
    
        return delta    
    
"""*********************用于计算期货期权的Black model************************"""

def dOne(s, k, r, t, v):
    """计算black模型中的d1"""
    d1 = (np.log(s / k) + 0.5 * v**2  * (t/252)) / (v * np.sqrt(t/252))
    return d1

#----------------------------------------------------------------------
def bsPrice(s, k, r, t, v, cp):
    """使用black模型计算期权的价格"""
    d1 = dOne(s, k, r, t, v)
    d2 = d1 - v * np.sqrt(t/252)

    price = cp * np.exp(-r * t/252) * (s * norm.cdf(cp * d1) - k * norm.cdf(cp * d2)) 
    return price

#----------------------------------------------------------------------
def bsDelta(s, k, r, t, v, cp): 
    """使用black模型计算期权的Delta"""
    d1 = dOne(s, k, r, t, v)
    delta = np.exp(-r * t/252) * cp * norm.cdf(cp * d1)
    return delta

#def pdfNorm(x):
#    """正态分布概率密度函数"""
#    y = 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)
#    return y 

def bsVega(s, k, r, t, v):
    """使用black模型计算期权的Vega"""
    d1 = dOne(s, k, r, t, v)
    result = s * np.sqrt(t/252) * np.exp(-r * t/252) * norm.pdf(d1)
    
    return result

def impliedVolatility(fPrice, fVega, price):
    
    TOLABS  = 1e-6
    MAXITER = 100
    
    sigma  = 0.3          # intial estimate   
    dSigma = 10 * TOLABS  # enter loop for the first time
    nIter  = 0
    while ((nIter < MAXITER) & (abs(dSigma) > TOLABS)):
        nIter = nIter + 1
        # Newton-Raphson method
        dSigma = (fPrice(sigma)-price)/fVega(sigma)
        sigma = sigma - dSigma
    
    if (nIter == MAXITER):
        warning('Newton-Raphson not converged')
    return sigma

def impliedVol(S0, K, r, T, cp, price):
    def fPrice(sigma):
        return bsPrice(S0, K, r, T, sigma, cp)
    def fVega(sigma):
        return bsVega(S0, K, r, T, sigma)
    impliedSigma  = impliedVolatility(fPrice,fVega,price)
    
    return impliedSigma

def DeltaHedgeFixedInterval(S0,K,r,T,sigma,M,N, Q, tick, commission, slippage, freq=1):
#固定时间对冲，频率单位为天, T单位为年，N为天数，即每天模拟出一个数据, N=T*365, Q为单位期权数量
    pricingEngine = MCpricing()
    greeks = Greeks()
    S = numRoundArr(pricingEngine.MCsimulation(S0,T,sigma,M,N), tick)
    
    M1 = 10000 
    N1 = 500
    optionPrice0 = pricingEngine.priceEuropeanCallMC(S0,K,r,T,sigma,M1,N1) * Q
    delta0 = greeks.delta(S0,K,r,T,sigma,M1,N1) * Q
    cashAccount0 = optionPrice0 - np.round(delta0) * S0 * (1 + slippage) * (1 + commission)
    
    deltaArr = np.zeros((M,N+1))
    optionPriceArr = np.zeros((M,N+1))
    cashAccountArr = np.zeros((M,N+1))
    
    deltaArr[:,0] = delta0
    optionPriceArr[:,0] = optionPrice0
    cashAccountArr[:,0] = cashAccount0
    
    
    for i in range(M):
        for j in range(N):
            delta = greeks.delta(S[i,j+1],K,r,T-(j+1),sigma,M1,N1) * Q
            optionPrice = pricingEngine.priceEuropeanCallMC(S[i,j+1],K,r,T-(j+1),sigma,M1,N1) * Q
            deltaSign = np.sign(delta-deltaArr[i,j])
            if j==N-1:
                cashAccount = cashAccountArr[i,j] + (optionPriceArr[i,j]-optionPrice)\
                                + np.round(deltaArr[i,j])*S[i,j+1]*(1-slippage)*(1-commission)
            else:
                cashAccount = cashAccountArr[i,j] + (optionPriceArr[i,j]-optionPrice)\
                                + np.round(deltaArr[i,j])*(S[i,j+1]-S[i,j])\
                                + deltaSign*S[i,j+1]*np.round(delta-deltaArr[i,j])*(1+deltaSign*slippage)*(1+deltaSign*commission)
                                
            deltaArr[i,j+1] = delta
            optionPriceArr[i,j+1] = optionPrice
            cashAccountArr[i,j+1] = cashAccount
            
    return cashAccountArr



#********************************************************************************************************************
def DeltaHedgeBSnoRoundDelta(S0,K,r,T,sigma,M,N,Q,C,tick, commission, slippage, cp=1, bs=-1):
#主要针对期权卖方   
#固定时间对冲，频率单位为天, T单位为年，N为天数，即每天模拟出一个数据, N=T*365
#call: cp=1, Put: cp=-1    买入期权：bs=1, 卖出期权：bs=-1
#tick为最小变动价位，Q为每手交易单位
#slippage为滑点（为tick的跳），commission为每手固定手续费

    pricingEngine = MCpricing()
    S = numRoundArr(pricingEngine.MCsimulation(S0,T,sigma,M,N), tick)
    
    optionPrice = bsPrice(S0,K,r,T,sigma,cp) * Q * C
    delta = bsDelta(S0,K,r,T,sigma,cp) * Q * C * bs
    

    cashAmount = -bs * optionPrice + delta * (S0 - cp * bs * slippage * tick) -  np.abs(delta) / Q * commission


    deltaArr = np.zeros((M,N+1))

    cashAccountArr = np.zeros((M,N+1))
 
    deltaArr[:,0] = delta

    cashAccountArr[:,0] = cashAmount
    
    for i in range(M):
        for j in range(N):
            if j==N-1:
                deltaSign = np.sign(deltaArr[i,j])
                cashAmount = cashAccountArr[i,j] * np.exp(r*1/252)\
                            - deltaArr[i,j] * (S[i,j+1] + deltaSign * slippage * tick)\
                            - np.abs(deltaArr[i,j]) / Q * commission\
                            + bs * np.maximum((S[i,j+1]-K)*cp,0) * Q * C
#                cashAmount = cashAccountArr[i,j] * np.exp(r*1/365)\
#                             - np.round(deltaArr[i,j])*S[i,j+1]*(1 + bs*slippage)\
#                             - np.abs(np.round(deltaArr[i,j])) / Q * commission\
#                             + bs * np.maximum(S[i,j+1]-K,0) * Q
            else:
                delta = bsDelta(S[i,j+1],K,r,T-(j+1),sigma,cp) * Q * C * bs
                deltaChange = delta-deltaArr[i,j]
                deltaSign = np.sign(deltaChange)

                cashAmount = cashAccountArr[i,j] * np.exp(r*1/252) \
                            + deltaChange * (S[i,j+1] - deltaSign*slippage*tick)\
                            - np.abs(deltaChange) / Q * commission

#                cashAmount = cashAccountArr[i,j] * np.exp(r*1/365) \
#                            + S[i,j+1] * np.round(deltaChange) * (1-deltaSign*slippage)\
#                            - np.abs(np.round(deltaChange)) / Q * commission                           
                deltaArr[i,j+1] = delta
           
            cashAccountArr[i,j+1] = cashAmount
    hedgeCost = np.mean(cashAccountArr[:,-1]) / Q / C
    return hedgeCost

#***************************BS model Hedge***********************************************    
def DeltaHedgeBS(S0,K,r,T,sigma,M,N,Q,C,tick, commission, slippage, cp=1, bs=-1):
#主要针对期权卖方   
#固定时间对冲，频率单位为天, T单位为年，N为天数，即每天模拟出一个数据, N=T*365
#call: cp=1, Put: cp=-1    买入期权：bs=1, 卖出期权：bs=-1
#tick为最小变动价位，Q为每手交易单位, C为手数
#slippage为滑点，commission为每手固定手续费
  
    pricingEngine = MCpricing()
    S = numRoundArr(pricingEngine.MCsimulation(S0,T,sigma,M,N), tick)
    
    optionPrice = bsPrice(S0,K,r,T,sigma,cp) * Q * C
    delta = bsDelta(S0,K,r,T,sigma,cp) * Q * C * bs
    


    cashAmount = -bs * optionPrice + np.round(delta) * (S0 - cp * bs * slippage * tick)\
                - np.abs(np.round(delta)) / Q * commission

    deltaArr = np.zeros((M,N+1))

    cashAccountArr = np.zeros((M,N+1))
 
    deltaArr[:,0] = delta

    cashAccountArr[:,0] = cashAmount
    
    for i in range(M):
        for j in range(N):
            if j==N-1:
#                cashAmount = cashAccountArr[i,j] * np.exp(r*1/365)\
#                            - deltaArr[i,j]*S[i,j+1]*(1 + bs*slippage) - np.abs(deltaArr[i,j]) / Q * commission\
#                            + bs * np.maximum(S[i,j+1]-K,0) * Q
                deltaSign = np.sign(deltaArr[i,j])
                cashAmount = cashAccountArr[i,j] * np.exp(r*1/252)\
                             - np.round(deltaArr[i,j])*(S[i,j+1] + deltaSign * slippage * tick)\
                             - np.abs(np.round(deltaArr[i,j])) / Q * commission\
                             + bs * np.maximum((S[i,j+1]-K)*cp,0) * Q * C
            else:
                delta = bsDelta(S[i,j+1],K,r,T-(j+1),sigma,cp) * Q * C * bs
                deltaChange = delta-deltaArr[i,j]
                deltaSign = np.sign(deltaChange)

#                cashAmount = cashAccountArr[i,j] * np.exp(r*1/365) \
#                            + S[i,j+1] * deltaChange * (1-deltaSign*slippage)\
#                            - np.abs(deltaChange) / Q * commission
 
                cashAmount = cashAccountArr[i,j] * np.exp(r*1/252) \
                            + np.round(deltaChange) * (S[i,j+1]-deltaSign*slippage*tick)\
                            - np.abs(np.round(deltaChange)) / Q * commission                           
                deltaArr[i,j+1] = delta
           
            cashAccountArr[i,j+1] = cashAmount
    
    hedgeCost = np.mean(cashAccountArr[:,-1]) / Q / C
    
    return hedgeCost

#****************************************************************************************************************************

class BidAsk(object):
    
    def __init__(self,code, tick, T, r):
#        self.code_tick = code_tick          #形如（'a',1）
        self.T = T
        self.r = r
#        self.code = code_tick[0]
#        self.tick = code_tick[1]
        self.code = code
        self.tick = tick
        codeSplit = re.split('\d',self.code)
        
#        w.start()
        self.EndDay = datetime.date.today() 
        BeginDay = self.EndDay - datetime.timedelta(days=365) * 2
        
#**********************根据主力连续合约计算波动率锥******************************
        mainContractLogReturn = FindChgDate.RevisedLogRet(codeSplit[0]+codeSplit[-1],BeginDay,self.EndDay)
        self.mainConvol = (mainContractLogReturn.rolling(window=self.T,center=False).std() * np.sqrt(252)).dropna()
        self.mainConvolstd = np.std(self.mainConvol)[0]
        self.todayVol = self.mainConvol.iloc[-1][0]
#        self.mainConCPrice = w.wsd(codeSplit[0]+codeSplit[-1], 'close', BeginDay, self.EndDay)
#        
#        mainConlogReturn = pd.Series(self.mainConCPrice.Data[0]).apply(np.log).diff(1)
#        self.mainConvol = (mainConlogReturn.rolling(window=self.T,center=False).std() * np.sqrt(252)).dropna()
#        self.mainConvolstd = np.std(self.mainConvol)

#**********************特定期货合约的波动率**************************************  
        
#        ContractLogReturn = FindChgDate.OriginalLogRetOneDate(self.code,self.EndDay,self.T)
#        self.todayVol = np.std(ContractLogReturn)[0] * np.sqrt(252)
        
      
          

#*******************************************************************************        
    def BidAskExtraVol(self, S0, K, param1=0.25, param2=0.5):
        

        moneyness = max(S0,K)/min(S0,K)
        ratio = self.mainConvolstd / self.todayVol
#   人为设定，为可变参数    
        fixedCoff1 = 0 * moneyness
        fixedCoff2 = param2 * self.mainConvolstd * moneyness
        fixedCoff3 = param1 * self.mainConvolstd * moneyness
        fixedCoff4 = self.mainConvolstd * moneyness
        
#        maxvol    = np.max(self.vol)
#        minvol    = np.min(self.vol)
#    ntpervol  = np.percentile(vol,90)
        self.sfpervol  = np.percentile(self.mainConvol,75)
        self.medianvol = np.percentile(self.mainConvol,50)
        self.tfpervol  = np.percentile(self.mainConvol,25)
#    tpervol   = np.percentile(vol,10)
        
        volList = self.mainConvol.iloc[:,0].tolist()
        volList.append(self.todayVol)
        
        todayVolPct = pd.Series(volList).rank().iloc[-1]/len(volList)
        
        if ratio > 0.1:
            
            if self.todayVol >= self.sfpervol:
                askExtraVol = fixedCoff1
                bidExtraVol = -fixedCoff2
        
            elif (self.todayVol < self.sfpervol) & (self.todayVol >= self.medianvol):
                askExtraVol = (2*todayVolPct-0.5)*(0.75-todayVolPct)/(0.75-0.5)*fixedCoff3
                bidExtraVol = -(2*todayVolPct-0.5)*(2-(0.75-todayVolPct)/(0.75-0.5))*fixedCoff3
        
            elif (self.todayVol < self.medianvol) & (self.todayVol >= self.tfpervol):
                askExtraVol = (-2*todayVolPct+1.5)*(2-(todayVolPct-0.25)/(0.5-0.25))*fixedCoff3
                bidExtraVol = -(-2*todayVolPct+1.5)*(todayVolPct-0.25)/(0.5-0.25)*fixedCoff3
        
            else:
                askExtraVol = fixedCoff2
                bidExtraVol = fixedCoff1
                
        else:
            
            if self.todayVol >= self.sfpervol:
                askExtraVol = fixedCoff1
                bidExtraVol = -fixedCoff4
        
            elif (self.todayVol < self.sfpervol) & (self.todayVol >= self.medianvol):
                askExtraVol = (2*todayVolPct-0.5)*(0.75-todayVolPct)/(0.75-0.5)*fixedCoff2
                bidExtraVol = -(2*todayVolPct-0.5)*(2-(0.75-todayVolPct)/(0.75-0.5))*fixedCoff2
        
            elif (self.todayVol < self.medianvol) & (self.todayVol >= self.tfpervol):
                askExtraVol = (-2*todayVolPct+1.5)*(2-(todayVolPct-0.25)/(0.5-0.25))*fixedCoff2
                bidExtraVol = -(-2*todayVolPct+1.5)*(todayVolPct-0.25)/(0.5-0.25)*fixedCoff2
        
            else:
                askExtraVol = fixedCoff4
                bidExtraVol = fixedCoff1
            
        return askExtraVol, bidExtraVol,self.todayVol


#    def BidAskPrice11(self,K,M,N,Q,C, commission, slippage, cp):
#        
#        S0 = self.ConCPrice.Data[0][-1]
#        sigma = np.mean(self.vol.iloc[-5:])
#        BSprice = bsPrice(S0,K,self.r,self.T, sigma, cp)
#        bidHedgeCost = DeltaHedgeBS(S0,K,self.r,self.T,sigma,M,N,Q,C,self.tick, commission, slippage, cp, -1)
#        askHedgeCost = DeltaHedgeBS(S0,K,self.r,self.T,sigma,M,N,Q,C,self.tick, commission, slippage, cp, 1)
#    
#        bidPrice = np.abs(bidHedgeCost) + bsPrice(S0,K,self.r,self.T, sigma+self.BidAskExtraVol()[0], cp)
#        askPrice = -np.abs(askHedgeCost) + bsPrice(S0,K,self.r,self.T, sigma+self.BidAskExtraVol()[1], cp)
#        
#        return bidPrice, askPrice, S0, BSprice, np.abs(bidHedgeCost),np.abs(askHedgeCost),\
#                self.BidAskExtraVol()[0],sigma,self.volstd, self.todayVol


    def BidAskPrice(self,S0, K, cp):   
#        S0 = w.wsq(self.code, "rt_last", func=DemoWSQCallback)
        sigma = self.todayVol
    
#        hedgeSigma = 45 * self.tick/S0
        self.hedgeSigma = 53 * self.tick / S0 + 0.015
        self.askExtraSigma = self.BidAskExtraVol(S0, K)[0]
        self.bidExtraSigma = self.BidAskExtraVol(S0, K)[1]
    
        self.askSigma = sigma + self.hedgeSigma + self.askExtraSigma
        self.bidSigma = sigma - self.hedgeSigma + self.bidExtraSigma
        
        askPrice = bsPrice(S0,K,self.r,self.T, self.askSigma, cp)
        bidPrice = bsPrice(S0,K,self.r,self.T, self.bidSigma, cp)
        
        return askPrice, bidPrice   
#    , bsPrice(S0,K,r,self.T, sigma, cp), hedgeSigma,\
#                bidExtraSigma,askExtraSigma,sigma,S0, bidSigma,askSigma,self.BidAskExtraVol(S0, K)[2],self.BidAskExtraVol(S0, K)[3]


    def calculation(self,S0,K):
        
        CA, CB = self.BidAskPrice(S0,K,1)
        PA, PB = self.BidAskPrice(S0,K,-1)
        
        return [[CA,CB,PA,PB],[self.tfpervol, self.medianvol, self.sfpervol,\
                self.todayVol],[self.askSigma,self.bidSigma,self.hedgeSigma,
                                 self.askExtraSigma, self.bidExtraSigma]]
    
    def code_to_name_load(self):
        
        f = open('CodeParseName.json')
        d = json.load(f)
       
        return d
    
    def QuoteOneSpecies(self, S0, K):
        
        quotedf = pd.DataFrame()
        ctype = 'C'
        ptype = 'P'
        expirationstr = '1M'
#        companyname = u'渤海融盛资本管理有限公司'

#        KSratio = [0.92, 0.97, 1, 1.03, 1.08]
#        
#        S0 = self.ConCPrice.Data[0][-1]
        
        parsedict =self.code_to_name_load()
        datelist = []   
        spotlist = []
        selllist = []
        buylist = []
        unitlist = []
        expirationlist = []
        excercisepricelist = []
        optiontypelist = []
        specieslist = []
#        companynamelist = []
        contractnamelist = []
        tfpervollist = []
        medianvollist = []
        sfpervollist = []
        todayVollist = [] 
        
        askVolList = []
        bidVolList = []
        hedgeVolList = []
        askExtraVolList = []
        bidExtraVolList = []
        
#        for i in range(len(KSratio)):
#            K = S0 * KSratio[i]
        CA, CB, PA, PB = self.calculation(S0,K)[0]
        tfpervol, medianvol, sfpervol, todayVol = self.calculation(S0,K)[1]
        askVol, bidVol, hedgeVol, askExtraVol, bidExtraVol = self.calculation(S0,K)[2]
        
        name_key = re.split('\d', self.code)[0]
        name = parsedict[name_key]
        
        buylist.append('%.4f %%' %(CB/S0*100))
        datelist.append(self.EndDay)
        selllist.append('%.4f %%' %(CA/S0*100))
        spotlist.append(S0)
        unitlist.append(self.tick)
        expirationlist.append(expirationstr)
        excercisepricelist.append(K)
        optiontypelist.append(ctype)
        specieslist.append(self.code)
#        companynamelist.append(companyname)
        contractnamelist.append(name)
        tfpervollist.append('%.4f %%' %(tfpervol*100))
        medianvollist.append('%.4f %%' %(medianvol*100))
        sfpervollist.append('%.4f %%' %(sfpervol*100))
        todayVollist.append('%.4f %%' %(todayVol*100))
        askVolList.append('%.4f %%' %(askVol*100))
        bidVolList.append('%.4f %%' %(bidVol*100))
        hedgeVolList.append('%.4f %%' %(hedgeVol*100))
        askExtraVolList.append('%.4f %%' %(askExtraVol*100))
        bidExtraVolList.append('%.4f %%' %(bidExtraVol*100))
        
        buylist.append('%.4f %%' %(PB/S0*100))
        datelist.append('')
        selllist.append('%.4f %%' %(PA/S0*100))
        spotlist.append('')
        unitlist.append('')
        expirationlist.append('')
        excercisepricelist.append('')
        optiontypelist.append(ptype)
        specieslist.append('')   
#        companynamelist.append(companyname)
        contractnamelist.append('')
        tfpervollist.append('')
        medianvollist.append('')
        sfpervollist.append('')
        todayVollist.append('')   
        askVolList.append('')
        bidVolList.append('')
        hedgeVolList.append('')
        askExtraVolList.append('')
        bidExtraVolList.append('')
        
        
#        quotedf[u'公司名称'] = companynamelist
        quotedf[u'报价日期'] = datelist
        quotedf[u'品种'] = contractnamelist
        quotedf[u'品种代码'] = specieslist
        quotedf[u'期权类型'] = optiontypelist
        quotedf[u'标的价格'] = spotlist
        quotedf[u'行权价'] = excercisepricelist
        quotedf[u'到期日/交易期限'] = expirationlist
        quotedf[u'最小交易单位'] = unitlist
        quotedf[u'买价'] = buylist
        quotedf[u'卖价'] = selllist
        
        quotedf[u''] = ''
        quotedf[u'25%分位波动率'] = tfpervollist
        quotedf[u'50%分位波动率'] = medianvollist
        quotedf[u'75%分位波动率'] = sfpervollist
        quotedf[u'今日波动率']    = todayVollist
        quotedf[u'买价波动率']    = bidVolList
        quotedf[u'卖价波动率']    = askVolList
        quotedf[u'对冲成本波动率'] = hedgeVolList
        quotedf[u'设定卖价溢出波动率'] = askExtraVolList
        quotedf[u'设定买价溢出波动率'] = bidExtraVolList
        
        return quotedf


        
    def quote_to_excel(self, bookname, df):
        filepath = 'D:\\simulation\\Quote\\'
        savename = bookname + '(' + self.EndDay.strftime('%Y-%m-%d') +  ')' + '.xlsx'
        filename = filepath + savename
        wb = xw.Book()
        sht = wb.sheets[0]
        sht.range('A1').options(pd.DataFrame, index = False).value = df
        sht.autofit()

        wb.save(filename)
        wb.close()
    
        
if __name__ == '__main__':
    w.start()
    code = ['CU1802.SHF','AL1802.SHF','ZN1802.SHF','NI1805.SHF',
            'RB1805.SHF','I1805.DCE','J1805.DCE','a1805.DCE',
            'c1805.DCE','M1805.DCE','p1805.DCE','RM805.CZC',
            'OI805.CZC','SR805.CZC','CF805.CZC','TA805.CZC',
            'JD1805.DCE','l1805.DCE','pp1805.DCE', 'ZC805.CZC', 
            'AU1806.SHF','RU1805.SHF','AG1806.SHF','Y1805.DCE',
            'TF1806.CFE','T1806.CFE','IH1801.CFE','IC1801.CFE',
            'IF1801.CFE'] 
    r  = 0.038
    l = len(code)
    
#    df = pd.DataFrame(columns = [u'公司名称',u'品种',u'品种代码',
#                                 u'期权类型',u'行权价',u'到期日/交易期限',
#                                 u'最小交易单位',u'买价',u'卖价',
#                                 u'标的价格',u'报价日期'])
    df = pd.DataFrame(columns = [u'报价日期',u'品种',u'品种代码',
                                 u'期权类型',u'标的价格',u'行权价',u'到期日/交易期限',
                                 u'最小交易单位',u'买价',u'卖价',u'',u'25%分位波动率',
                                 u'50%分位波动率',u'75%分位波动率',u'今日波动率',
                                 u'买价波动率',u'卖价波动率',u'对冲成本波动率',
                                 u'设定卖价溢出波动率',u'设定买价溢出波动率'])
    
    T = w.tdayscount(datetime.date.today(),datetime.date.today()+relativedelta(months=1), "").Data[0][0]
    for i in range(l):
        OriTickData = w.wss(code[i], "mfprice").Data[0][0]
        tick = float(re.search(r'\d+(\.\d+)?', OriTickData).group(0))

        S0 = w.wsq(code[i], "rt_last").Data[0][0]
        K = S0
        
        a = BidAsk(code[i],tick,T,r)

        df1 = a.QuoteOneSpecies(S0, K)
        df = df.append(df1)
        print (code[i])
        
#    bookname = u'场外期权报价-渤海融盛'
    bookname = u'场外期权报价'
    a.quote_to_excel(bookname, df)    



