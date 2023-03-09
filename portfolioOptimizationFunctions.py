# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:33:03 2020

@author: jamal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as sco


def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, _freq):
    results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate, _freq)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, _freq)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix, _freq)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=mean_returns.index,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix, _freq)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix, _freq)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=mean_returns.index,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.2, 100)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target, _freq)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)    
    
    out = pd.concat([pd.DataFrame([p['fun'] for p in efficient_portfolios]), pd.DataFrame(target)],axis=1)
    out.columns = ['vol','ret']
    return out

def portfolio_annualised_performance(weights, mean_returns, cov_matrix, _freq):
    returns = np.sum(mean_returns*weights ) * _freq
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(_freq)
    return std, returns


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, _freq):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns.index))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix, _freq)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate, _freq):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix, _freq)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, _freq):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate, _freq)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix, _freq):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix, _freq)[0]

def min_variance(mean_returns, cov_matrix, _freq):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, _freq)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target, _freq):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, _freq)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix, _freq)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1.0) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range, _freq):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret, _freq))
    return efficients
