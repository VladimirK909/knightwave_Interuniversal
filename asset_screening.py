import pandas as pd
import pypfopt
import pypfopt.expected_returns
import numpy as np
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

def calculate_sharpe_ratio(df_returns, risk_free_rate=0.0):
    excess_returns = df_returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio  # Returns a Series, indexed by asset name


def calculate_sortino_ratio(returns: pd.DataFrame, target=0.0):
    """
    returns: DataFrame where each column is an asset, and each row is a time period's return
    target: minimum acceptable return (usually 0 or risk-free rate)
    """
    excess_return = returns.mean() - target
    downside_diff = returns - target
    downside = downside_diff[downside_diff < 0].fillna(0)
    downside_std = np.sqrt((downside ** 2).mean(axis=0))  # fix is here
    sortino_ratio = excess_return / downside_std
    return sortino_ratio
