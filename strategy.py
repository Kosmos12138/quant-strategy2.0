# -*- coding: utf-8 -*-
"""
多因子选股+夏普比率最大化组合优化策略
"""

# 初始化函数
def initialize(context):
    # 设置回测参数
    set_params(context)
    # 设置交易费用和滑点
    set_slippage_and_commission(context)
    # 每月第1个交易日调仓
    run_monthly(before_market_open, 1, time='before_open')

def set_params(context):
    """设置参数"""
    # 股票池：沪深300成分股
    g.index = '000300.XSHG'
    # 因子数量控制
    g.selected_factors = 8   # LASSO筛选后的因子数
    # 组合优化参数
    g.max_stocks = 10        # 最大持仓股票数
    g.min_weight = 0.01      # 最小权重
    g.max_weight = 0.40      # 最大权重
    # 设置调仓日
    g.rebalance_day = 1
    
def set_slippage_and_commission(context):
    """设置交易成本"""
    set_slippage(FixedSlippage(0.002))  # 固定滑点0.2%
    set_order_cost(OrderCost(
        open_tax=0,           # 买入印花税
        close_tax=0.001,      # 卖出印花税0.1%
        open_commission=0.0003,  # 买入佣金
        close_commission=0.0003, # 卖出佣金
        close_today_commission=0, # 今平佣金
        min_commission=5       # 最低佣金
    ), type='stock')

# ==================== 辅助函数 ====================

def calculate_rsi(prices, period=14):
    """计算RSI"""
    import pandas as pd
    import numpy as np
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_sharpe_ratio(prices, period=60, risk_free_rate=0.03):
    """计算夏普比率"""
    returns = prices.pct_change().dropna().iloc[-period:]
    if len(returns) < 20:
        return pd.Series(0.0, index=prices.columns)
    
    daily_risk_free = risk_free_rate / 252
    excess_returns = returns - daily_risk_free
    return excess_returns.mean() / excess_returns.std()

def get_roe(stock_list, current_dt):
    """获取ROE"""
    q = query(
        indicator.code,
        indicator.roe
    ).filter(indicator.code.in_(stock_list))
    df = get_fundamentals(q, date=current_dt)
    if df.empty:
        return pd.Series(0.1, index=stock_list)  # 默认值
    return pd.Series(df['roe'].values, index=df['code'])

def get_roa(stock_list, current_dt):
    """获取ROA"""
    q = query(
        indicator.code,
        indicator.roa
    ).filter(indicator.code.in_(stock_list))
    df = get_fundamentals(q, date=current_dt)
    if df.empty:
        return pd.Series(0.05, index=stock_list)  # 默认值
    return pd.Series(df['roa'].values, index=df['code'])

def get_revenue_growth(stock_list, current_dt):
    """获取营收增长率"""
    q = query(
        indicator.code,
        indicator.inc_revenue_year_on_year
    ).filter(indicator.code.in_(stock_list))
    df = get_fundamentals(q, date=current_dt)
    if df.empty:
        return pd.Series(0.1, index=stock_list)  # 默认值
    return pd.Series(df['inc_revenue_year_on_year'].values, index=df['code'])

# ==================== 因子计算函数 ====================

def calculate_factors(security_list, current_dt):
    """计算多因子 - 简化版本，避免复杂计算"""
    import pandas as pd
    import numpy as np
    
    # 获取120个交易日的价格数据
    prices = get_price(
        security_list, 
        end_date=current_dt,
        count=120, 
        fields=['close', 'volume', 'money'],
        fq='pre'
    )
    
    close_prices = prices['close']
    volumes = prices['volume']
    amounts = prices['money']
    
    # 只选择有足够数据的股票
    valid_stocks = close_prices.columns[close_prices.count() > 60].tolist()
    if not valid_stocks:
        return pd.DataFrame()
    
    factors_dict = {}
    
    try:
        # 1. 动量类因子
        if len(close_prices) >= 20:
            factors_dict['MOM_1M'] = close_prices.iloc[-20] / close_prices.iloc[-1] - 1
        if len(close_prices) >= 60:
            factors_dict['MOM_3M'] = close_prices.iloc[-60] / close_prices.iloc[-1] - 1
        if len(close_prices) >= 120:
            factors_dict['MOM_6M'] = close_prices.iloc[-120] / close_prices.iloc[-1] - 1
        
        # 2. 波动率类因子
        factors_dict['VOL_20D'] = close_prices.pct_change().rolling(20).std().iloc[-1]
        factors_dict['VOL_60D'] = close_prices.pct_change().rolling(60).std().iloc[-1]
        
        # 3. 成交量类因子
        if len(volumes) >= 20:
            volume_mean = volumes.rolling(20).mean().iloc[-1]
            volume_ratio = volumes.iloc[-1] / volume_mean
            volume_ratio = volume_ratio.replace([np.inf, -np.inf], 1)
            factors_dict['VOLUME_RATIO'] = volume_ratio
            
            amount_mean = amounts.rolling(20).mean().iloc[-1]
            factors_dict['AMOUNT_20D'] = amount_mean
        
        # 4. 技术指标因子
        if len(close_prices) >= 20:
            ma_20 = close_prices.rolling(20).mean().iloc[-1]
            factors_dict['MA_RATIO'] = close_prices.iloc[-1] / ma_20
        
        # 5. 估值类因子
        fundamentals = get_fundamentals(
            query(
                valuation.code,
                valuation.pe_ratio,
                valuation.pb_ratio,
                valuation.ps_ratio,
                valuation.market_cap
            ).filter(
                valuation.code.in_(valid_stocks)
            ),
            date=current_dt
        )
        
        if not fundamentals.empty:
            factors_dict['PE'] = pd.Series(
                fundamentals['pe_ratio'].values,
                index=fundamentals['code']
            ).apply(lambda x: np.log(x) if x > 0 else np.log(20))
            
            factors_dict['PB'] = pd.Series(
                fundamentals['pb_ratio'].values,
                index=fundamentals['code']
            ).apply(lambda x: np.log(x) if x > 0 else np.log(2))
            
            factors_dict['PS'] = pd.Series(
                fundamentals['ps_ratio'].values,
                index=fundamentals['code']
            )
            
            factors_dict['LOG_MCAP'] = pd.Series(
                fundamentals['market_cap'].values,
                index=fundamentals['code']
            ).apply(lambda x: np.log(x) if x > 0 else np.log(1e9))
        
        # 6. 质量类因子
        factors_dict['ROE'] = get_roe(valid_stocks, current_dt)
        factors_dict['ROA'] = get_roa(valid_stocks, current_dt)
        factors_dict['GROWTH'] = get_revenue_growth(valid_stocks, current_dt)
        
        # 7. 反转类因子
        if len(close_prices) >= 5:
            factors_dict['REV_1W'] = -1 * (close_prices.iloc[-5] / close_prices.iloc[-1] - 1)
        if len(close_prices) >= 20:
            factors_dict['REV_1M'] = -1 * (close_prices.iloc[-20] / close_prices.iloc[-1] - 1)
        
        # 8. 风险调整收益因子
        sharpe = calculate_sharpe_ratio(close_prices, period=60)
        factors_dict['SHARPE_60D'] = sharpe
        
        # 转换为DataFrame
        factor_df = pd.DataFrame(factors_dict)
        
        # 处理缺失值
        factor_df = factor_df.fillna(factor_df.median())
        factor_df = factor_df.replace([np.inf, -np.inf], 0)
        
        return factor_df
        
    except Exception as e:
        log.error(f"计算因子时出错: {e}")
        return pd.DataFrame()

# ==================== LASSO特征筛选 ====================

def lasso_feature_selection(factor_data, future_returns, n_features=10):
    """使用LASSO进行因子筛选"""
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    
    try:
        # 1. 数据清洗
        factor_data_clean = factor_data.dropna()
        future_returns_aligned = future_returns.reindex(factor_data_clean.index).dropna()
        factor_data_clean = factor_data_clean.reindex(future_returns_aligned.index)
        
        if len(factor_data_clean) < 30 or len(future_returns_aligned) < 30:
            # 数据不足，返回所有因子
            return factor_data.columns.tolist()[:n_features], pd.Series()
        
        # 2. 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(factor_data_clean)
        y = future_returns_aligned.values
        
        # 3. LASSO回归
        lasso_cv = LassoCV(cv=5, max_iter=5000, n_alphas=30, random_state=42)
        lasso_cv.fit(X_scaled, y)
        
        # 4. 获取重要因子
        coef = pd.Series(lasso_cv.coef_, index=factor_data_clean.columns)
        selected_factors = coef[coef.abs() > 0.01].index.tolist()
        
        # 5. 如果选择的因子不合适，调整
        if len(selected_factors) > n_features:
            # 按系数绝对值排序
            selected_factors = coef.abs().sort_values(ascending=False).head(n_features).index.tolist()
        elif len(selected_factors) < 3:
            # 用与收益率相关性最高的因子补充
            correlations = factor_data_clean.apply(lambda x: np.corrcoef(x, y)[0,1] if len(x.dropna()) > 10 else 0)
            selected_factors = correlations.abs().sort_values(ascending=False).head(n_features).index.tolist()
        
        return selected_factors, coef
        
    except Exception as e:
        log.warning(f"LASSO筛选失败: {e}")
        # 返回默认因子
        default_factors = ['MOM_1M', 'MOM_3M', 'VOL_20D', 'PE', 'PB', 'ROE', 'GROWTH', 'VOLUME_RATIO', 'SHARPE_60D', 'MA_RATIO']
        available_factors = [f for f in default_factors if f in factor_data.columns]
        return available_factors[:n_features], pd.Series()

# ==================== 股票打分模型 ====================

def score_stocks(factor_data, selected_factors):
    """基于筛选后的因子进行综合打分"""
    import pandas as pd
    import numpy as np
    
    try:
        scores = pd.Series(0.0, index=factor_data.index)
        
        # 因子方向定义
        factor_directions = {
            'MOM_1M': 1, 'MOM_3M': 1, 'MOM_6M': 1,
            'VOL_20D': -1, 'VOL_60D': -1,
            'VOLUME_RATIO': 1, 
            'PE': -1, 'PB': -1, 'PS': -1,
            'ROE': 1, 'ROA': 1, 'GROWTH': 1,
            'REV_1W': 1, 'REV_1M': 1,
            'SHARPE_60D': 1,
            'MA_RATIO': 1,
            'LOG_MCAP': -1  # 小市值效应
        }
        
        for factor in selected_factors:
            if factor in factor_data.columns:
                factor_values = factor_data[factor]
                
                # 处理异常值
                factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
                factor_values = factor_values.fillna(factor_values.median())
                
                # 排名标准化 (0-1)
                factor_rank = factor_values.rank(pct=True)
                
                # 根据因子方向调整
                direction = factor_directions.get(factor, 1)
                if direction > 0:
                    scores = scores.add(factor_rank, fill_value=0)
                else:
                    scores = scores.add(1 - factor_rank, fill_value=0)
        
        # 归一化到0-100分
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min()) * 100
        else:
            scores = scores * 0 + 50  # 所有股票给50分
        
        return scores.sort_values(ascending=False)
        
    except Exception as e:
        log.error(f"股票打分失败: {e}")
        return pd.Series(50, index=factor_data.index)

# ==================== 组合优化 ====================

def optimize_portfolio(context, selected_stocks, factor_scores):
    """组合优化 - 使用简化的方法"""
    import pandas as pd
    import numpy as np
    
    try:
        # 1. 获取历史收益率
        hist_prices = get_price(
            selected_stocks,
            end_date=context.current_dt,
            count=120,
            fields=['close'],
            fq='pre'
        )['close']
        
        if hist_prices.empty or len(hist_prices) < 60:
            return equal_weight_fallback(selected_stocks)
        
        # 2. 计算收益率和协方差矩阵
        returns = hist_prices.pct_change().dropna()
        if len(returns) < 30:
            return equal_weight_fallback(selected_stocks)
        
        # 3. 计算预期收益率（使用历史均值）
        expected_returns = returns.mean()
        
        # 4. 计算协方差矩阵
        cov_matrix = returns.cov()
        
        # 5. 使用简化的均值-方差优化（最大化夏普比率）
        n = len(selected_stocks)
        
        # 初始权重（基于因子得分）
        if factor_scores is not None and not factor_scores.empty:
            # 确保factor_scores是Series且包含所有股票
            scores = []
            for stock in selected_stocks:
                if stock in factor_scores.index:
                    scores.append(factor_scores[stock])
                else:
                    scores.append(1.0)  # 默认值
            init_weights = np.array(scores)
            init_weights = init_weights / init_weights.sum()
        else:
            init_weights = np.ones(n) / n
        
        # 约束条件：权重和为1，单个权重在[min_weight, max_weight]之间
        from scipy.optimize import minimize
        
        def negative_sharpe(weights):
            """负夏普比率（要最小化）"""
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol < 1e-10:
                return -1000  # 高回报表示低质量
            return -port_return / port_vol  # 负号因为要最小化
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # 边界条件
        bounds = [(g.min_weight, g.max_weight) for _ in range(n)]
        
        # 优化
        result = minimize(
            negative_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            # 处理优化结果
            weights = result.x
            optimized_weights = {}
            
            for i, stock in enumerate(selected_stocks):
                if weights[i] > g.min_weight * 0.5:  # 稍微放宽阈值
                    optimized_weights[stock] = float(weights[i])  # 确保是float类型
            
            # 归一化 - 这里需要确保值是数值类型
            if optimized_weights:  # 如果字典不为空
                total_weight = 0.0
                # 先计算总和
                for weight in optimized_weights.values():
                    total_weight += float(weight)
                
                # 修正：确保 total_weight 是数值类型
                total_weight = float(total_weight)
                
                # 现在可以安全比较了
                if total_weight > 0:
                    optimized_weights = {k: float(v)/total_weight for k, v in optimized_weights.items()}
            
            # 限制持仓数量
            if len(optimized_weights) > g.max_stocks:
                # 按权重排序，保留最大的g.max_stocks个
                sorted_items = sorted(optimized_weights.items(), key=lambda x: float(x[1]), reverse=True)
                optimized_weights = dict(sorted_items[:g.max_stocks])
                # 重新归一化
                if optimized_weights:
                    total_weight = sum(float(v) for v in optimized_weights.values())
                    total_weight = float(total_weight)  # 确保是float
                    if total_weight > 0:
                        optimized_weights = {k: float(v)/total_weight for k, v in optimized_weights.items()}
            
            if optimized_weights:
                max_weight_val = max(float(v) for v in optimized_weights.values())
                log.info(f"优化成功，选择{len(optimized_weights)}只股票，最大权重{max_weight_val:.3f}")
            else:
                log.warning("优化后无有效持仓，使用等权重")
                return equal_weight_fallback(selected_stocks)
            
            return optimized_weights
        else:
            log.warning("优化失败，使用等权重")
            return equal_weight_fallback(selected_stocks)
            
    except Exception as e:
        log.error(f"组合优化出错: {e}")
        import traceback
        log.error(traceback.format_exc())  # 打印详细错误信息
        return equal_weight_fallback(selected_stocks)

# ==================== 主调仓逻辑 ====================

def before_market_open(context):
    """每月调仓前执行"""
    current_date = context.current_dt.date()
    
    # 检查是否为调仓日
    if current_date.day != g.rebalance_day:
        return
    
    log.info(f"========== 调仓日：{current_date} ==========")
    
    # 1. 获取股票池（沪深300成分股）
    stock_pool = get_index_stocks(g.index, date=context.current_dt)
    if len(stock_pool) < 50:
        log.warning("股票池数量不足，跳过调仓")
        return
    
    log.info(f"初始股票池数量：{len(stock_pool)}")
    
    # 2. 计算因子
    log.info("开始计算多因子...")
    factor_data = calculate_factors(stock_pool, context.current_dt)
    
    if factor_data.empty or len(factor_data) < 50:
        log.warning("因子数据不足，跳过调仓")
        return
    
    log.info(f"有效因子数据股票数：{len(factor_data)}")
    
    # 3. 获取未来20个交易日的收益率作为标签
    log.info("获取未来收益率标签...")
    try:
        future_prices = get_price(
            factor_data.index.tolist(),
            end_date=context.current_dt,
            count=40,  # 20个交易日用于计算未来收益
            fields=['close'],
            fq='pre'
        )['close']
        
        if len(future_prices) >= 20:
            # 计算未来20个交易日的收益率
            future_returns = (future_prices.iloc[-1] / future_prices.iloc[0] - 1)
        else:
            # 数据不足，使用历史波动率作为替代
            log.warning("未来价格数据不足，使用历史波动率替代")
            hist_returns = get_price(
                factor_data.index.tolist(),
                end_date=context.current_dt,
                count=60,
                fields=['close'],
                skip_paused=True,
                fq='pre'
            )['close'].pct_change().std()
            future_returns = -hist_returns  # 负相关：波动率越高，预期收益越低
    except Exception as e:
        log.warning(f"获取未来收益率失败: {e}")
        # 使用随机值
        future_returns = pd.Series(np.random.randn(len(factor_data)), index=factor_data.index)
    
    # 4. LASSO特征筛选
    log.info("进行LASSO特征筛选...")
    selected_factors, coef = lasso_feature_selection(
        factor_data, future_returns, g.selected_factors
    )
    
    log.info(f"筛选出{len(selected_factors)}个有效因子")
    if not coef.empty:
        log.info(f"重要因子系数: {coef[coef.abs() > 0.01].to_dict()}")
    
    # 5. 股票打分
    log.info("计算股票综合得分...")
    stock_scores = score_stocks(factor_data, selected_factors)
    
    if stock_scores.empty:
        log.warning("股票打分为空，跳过调仓")
        return
    
    # 6. 初选股票（按打分排序）
    top_n = min(50, len(stock_scores) // 2)
    top_stocks = stock_scores.head(top_n).index.tolist()
    log.info(f"初选{len(top_stocks)}只股票，最高分{stock_scores.max():.2f}，最低分{stock_scores.min():.2f}")
    
    # 7. 组合优化
    log.info("进行组合优化（夏普最大化）...")
    optimized_weights = optimize_portfolio(context, top_stocks, stock_scores)
    
    # 8. 执行调仓
    log.info(f"优化后持仓{len(optimized_weights)}只股票")
    execute_rebalance(context, optimized_weights)

def execute_rebalance(context, target_weights):
    """执行调仓"""
    # 卖出不在目标持仓中的股票
    for stock in context.portfolio.positions:
        if stock not in target_weights:
            order_target_value(stock, 0)
            log.info(f"卖出 {stock}")
    
    # 调整目标持仓
    total_value = context.portfolio.total_value
    
    for stock, weight in target_weights.items():
        target_value = total_value * weight
        current_value = context.portfolio.positions.get(stock, 0).value
        
        # 只调整差异较大的持仓
        if abs(target_value - current_value) > total_value * 0.01:  # 差异大于1%才调整
            order_target_value(stock, target_value)
            if target_value > current_value:
                log.info(f"买入 {stock}: {weight:.2%}")
            elif target_value < current_value:
                log.info(f"卖出 {stock}: 减少{((current_value - target_value)/current_value):.1%}")

# ==================== 每日监控 ====================

def handle_data(context, data):
    """每日执行"""
    # 可以在这里添加每日监控逻辑
    pass

# ==================== 回测后分析 ====================

def after_trading_end(context):
    """每日收盘后执行"""
    # 记录每日持仓
    pass

# 运行回测
if __name__ == "__main__":
    # 这些代码在聚宽环境中不需要
    pass