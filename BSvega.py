# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:24:24 2017

@author: BHRS-ZY-PC
"""
import numpy as np
from scipy.stats import norm


def vegaEuropeanCall(S0,K,r,T,sigma):
    """
    Vega of a European call option
    
    INPUT:
       S0 : Initial value of the underlying asset
        K : Strike 
        r : Risk-free interest rate 
        T : Time to expiry 
    sigma : Volatility 

    OUTPUT:
     vega : Vega of the option in the Black-Scholes model  
    """
    discountedStrike = np.exp(-r * T) * K
    totalVolatility = sigma * np.sqrt(T)
    d_plus = np.log(S0 / discountedStrike) / totalVolatility + .5 * totalVolatility
    return S0 * np.sqrt(T) * norm.pdf(d_plus)