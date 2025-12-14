# quant-strategy2.0
multi-factor strategy &amp; JoinQuant's portfolio optimizer
# Multi-Factor Stock Selection Strategy with Sharpe Ratio Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Framework: JQ Research](https://img.shields.io/badge/framework-JQ%20Research-green.svg)](https://www.joinquant.com/)

A comprehensive quantitative trading strategy implementing multi-factor stock selection with LASSO feature selection and Sharpe ratio maximization portfolio optimization, designed for the JQ environment.

# Strategy Overview

This strategy combines multiple alpha factors across different categories to select stocks from the CSI 300 index, optimizes factor selection using LASSO regression, and constructs portfolios that maximize the Sharpe ratio.

# Key Features

- Multi-Factor Framework: 8+ factors across momentum, valuation, quality, technical, and reversal categories
- Intelligent Feature Selection: LASSO regression for optimal factor combination
- Portfolio Optimization: Sharpe ratio maximization with realistic constraints
- Comprehensive Analysis: IC analysis, factor correlation, and performance metrics
- Professional Visualization: English charts for strategy evaluation
- Practical Implementation: Designed for JQ Research platform with real market constraints

# Factor Categories

| Category | Factors | Description |
|----------|---------|-------------|
| Momentum | MOM_1M, MOM_3M, MOM_6M | Price momentum over 1, 3, and 6 months |
| Valuation| PE, PB, PS, LOG_MCAP | Valuation ratios and market cap |
| Quality  | ROE, ROA, GROWTH | Profitability and growth metrics |
| Technical| VOL_20D, VOL_60D, MA_RATIO, VOLUME_RATIO | Volatility and technical indicators |
| Reversal | REV_1W, REV_1M | Short-term price reversal |
| Risk-Adjusted | SHARPE_60D | 60-day Sharpe ratio |

# Prerequisites

- JoinQuant Research account ( https://www.joinquant.com/)
- Basic Python knowledge
- Understanding of quantitative finance concepts


# Configuration
python
class Config:
    # Backtest period
    start_date = '2023-01-01'
    end_date = '2025-12-10'
    
    # Stock universe
    benchmark = '000300.XSHG'  # CSI 300 Index
    universe = '000300.XSHG'
    
    # Factor parameters
    selected_factors = 8        # Number of factors after LASSO selection
    
    # Portfolio parameters
    max_stocks = 10             # Maximum holdings
    min_weight = 0.01           # Minimum weight per stock
    max_weight = 0.40           # Maximum weight per stock
    
    # Rebalancing
    rebalance_freq = 'M'        # Monthly rebalancing
    
    # IC analysis
    ic_lookback = 20            # IC calculation period
    ic_rolling_window = 12      # Rolling IC window

# Output and Visualizations
The strategy generates comprehensive visualizations:

# 1. Factor IC Analysis
- IC Time Series and Rank IC
- IC Distribution Histograms
- Average IC by Factor
- Rolling IC Mean

# 2. Factor Correlation
- Correlation Heatmap
- Factor Clustering
- Multicollinearity Analysis

# 3. Backtest Performance
- Cumulative Return vs Benchmark
- Monthly Return Distribution
- Annual Performance
- Rolling Risk Metrics

# 4. Portfolio Analysis
- Weight Distribution
- Turnover Analysis
- Sector Exposure
- Risk Decomposition

# Happy quant trading!! ~