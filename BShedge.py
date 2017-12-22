# encoding: UTF-8

"""
基于Black-Scholes模型，适用于欧式期权定价。

变量定义
s: 标的物价格
k: 行权价
r: 无风险利率
t: 剩余到期时间（年）
v: 隐含波动率
cp: 期权类型（看涨+1 看跌-1）
"""
from __future__ import division

from math import exp, sqrt, pow, log, erf
from scipy.stats import norm
import numpy as np
from numpy import random


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


#----------------------------------------------------------------------
def dOne(s, k, r, t, v):
    """计算bs模型中的d1"""
    return (log(s / k) + (0.5 * pow(v, 2) + r) * t) / (v * sqrt(t))
    
#----------------------------------------------------------------------
def bsPrice(s, k, r, t, v, cp):
    """使用bs模型计算期权的价格"""
    d1 = dOne(s, k, r, t, v)
    d2 = d1 - v * sqrt(t)
    price = cp * (s * norm.cdf(cp * d1) - k * exp(-r * t) * norm.cdf(cp * d2))
    return price

#----------------------------------------------------------------------
def bsDelta(s, k, r, t, v, cp): 
    """使用bs模型计算期权的Delta"""
    d1 = dOne(s, k, r, t, v)
    delta = cp * norm.cdf(cp * d1)
    return delta
    
#----------------------------------------------------------------------
def bsGamma(s, k, r, t, v, cp): 
    """使用bs模型计算期权的Gamma"""
    d1 = dOne(s, k, r, t, v)
    gamma = norm.pdf(d1) / (s * v * sqrt(t))
    return gamma

#----------------------------------------------------------------------
def bsVega(s, k, r, t, v, cp): 
    """使用bs模型计算期权的Vega"""
    d1 = dOne(s, k, r, t, v)
    vega = s * norm.pdf(d1) * sqrt(t)
    return vega

#----------------------------------------------------------------------
def bsTheta(s, k, r, t, v, cp): 
    """使用bs模型计算期权的Theta"""
    d1 = dOne(s, k, r, t, v)
    d2 = d1 - v * sqrt(t)
    theta = (-(s * v * norm.pdf(d1))/(2 * sqrt(t)) 
             - cp * r * k * exp(-r * t) * norm.cdf(cp * d2))
    return theta

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