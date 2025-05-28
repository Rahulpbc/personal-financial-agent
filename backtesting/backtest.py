"""
Backtesting module for portfolio strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Class for storing backtest results"""
    portfolio_value_history: pd.Series
    benchmark_history: pd.Series
    trades: List[Dict[str, Any]]
    metrics: Dict[str, float]
    start_date: datetime
    end_date: datetime
    tickers: List[str]
    initial_weights: Dict[str, float]
    final_weights: Dict[str, float]

class PortfolioBacktester:
    """Backtester for portfolio strategies"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the backtester
        
        Args:
            random_seed: Random seed for reproducibility (NFR-3)
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        logger.info(f"Portfolio backtester initialized with random seed {random_seed}")
    
    def backtest_portfolio(
        self,
        prices_df: pd.DataFrame,
        initial_weights: Dict[str, float],
        benchmark_ticker: str = "SPY",
        rebalance_frequency: str = "W",  # W: weekly, M: monthly, Q: quarterly
        initial_capital: float = 10000.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Backtest a portfolio strategy
        
        Args:
            prices_df: DataFrame with price data (index=dates, columns=tickers)
            initial_weights: Initial portfolio weights (ticker -> weight)
            benchmark_ticker: Ticker symbol for benchmark
            rebalance_frequency: Frequency for portfolio rebalancing
            initial_capital: Initial capital for the portfolio
            start_date: Start date for the backtest
            end_date: End date for the backtest
            
        Returns:
            BacktestResult object with backtest results
        """
        try:
            # Filter date range if provided
            if start_date is not None:
                prices_df = prices_df[prices_df.index >= start_date]
            if end_date is not None:
                prices_df = prices_df[prices_df.index <= end_date]
            
            # Ensure we have at least 6 months of data (FR-9)
            if len(prices_df) < 120:  # Approximately 6 months of trading days
                logger.warning(f"Limited price data available: {len(prices_df)} days, recommended at least 120 days")
            
            # Get actual start and end dates from the data
            start_date = prices_df.index[0]
            end_date = prices_df.index[-1]
            
            # Make sure all tickers in initial_weights are in prices_df
            tickers = list(initial_weights.keys())
            for ticker in tickers:
                if ticker not in prices_df.columns:
                    logger.warning(f"Ticker {ticker} not found in price data, removing from backtest")
                    initial_weights.pop(ticker)
            
            # Normalize weights to sum to 1
            weight_sum = sum(initial_weights.values())
            if weight_sum != 1.0:
                for ticker in initial_weights:
                    initial_weights[ticker] /= weight_sum
            
            # Initialize portfolio
            portfolio = pd.Series(initial_capital, index=prices_df.index)
            cash = initial_capital
            holdings = {ticker: 0 for ticker in initial_weights}
            
            # Initialize benchmark
            if benchmark_ticker in prices_df.columns:
                benchmark = prices_df[benchmark_ticker] / prices_df[benchmark_ticker].iloc[0] * initial_capital
            else:
                logger.warning(f"Benchmark ticker {benchmark_ticker} not found in price data, using equal-weighted portfolio as benchmark")
                # Use equal-weighted portfolio as benchmark
                equal_weights = {ticker: 1/len(prices_df.columns) for ticker in prices_df.columns}
                benchmark = pd.Series(0.0, index=prices_df.index)
                for ticker, weight in equal_weights.items():
                    benchmark += prices_df[ticker] / prices_df[ticker].iloc[0] * initial_capital * weight
            
            # Initialize trade log
            trades = []
            
            # Generate rebalance dates
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_frequency)
            
            # Run backtest
            for i, date in enumerate(prices_df.index):
                # Get current prices
                current_prices = prices_df.loc[date]
                
                # Calculate current portfolio value
                portfolio_value = cash
                for ticker, shares in holdings.items():
                    if ticker in current_prices:
                        portfolio_value += shares * current_prices[ticker]
                
                # Record portfolio value
                portfolio[date] = portfolio_value
                
                # Check if we need to rebalance
                if date in rebalance_dates:
                    logger.info(f"Rebalancing portfolio on {date.strftime('%Y-%m-%d')}")
                    
                    # Calculate target allocation
                    target_allocation = {ticker: portfolio_value * weight for ticker, weight in initial_weights.items()}
                    
                    # Calculate current allocation
                    current_allocation = {ticker: holdings.get(ticker, 0) * current_prices.get(ticker, 0) for ticker in initial_weights}
                    
                    # Calculate trades needed
                    for ticker in initial_weights:
                        if ticker in current_prices:
                            target_value = target_allocation[ticker]
                            current_value = current_allocation[ticker]
                            
                            # Calculate trade
                            trade_value = target_value - current_value
                            trade_shares = trade_value / current_prices[ticker]
                            
                            # Execute trade
                            if abs(trade_shares) > 0.0001:  # Avoid tiny trades
                                # Update holdings and cash
                                holdings[ticker] = holdings.get(ticker, 0) + trade_shares
                                cash -= trade_value
                                
                                # Log trade
                                trade = {
                                    "date": date.strftime("%Y-%m-%d"),
                                    "ticker": ticker,
                                    "action": "buy" if trade_shares > 0 else "sell",
                                    "shares": abs(trade_shares),
                                    "price": current_prices[ticker],
                                    "value": abs(trade_value),
                                    "reason": "Regular rebalancing"
                                }
                                trades.append(trade)
                                
                                logger.debug(f"Trade: {trade}")
            
            # Calculate final weights
            final_prices = prices_df.iloc[-1]
            final_portfolio_value = cash
            for ticker, shares in holdings.items():
                if ticker in final_prices:
                    final_portfolio_value += shares * final_prices[ticker]
            
            final_weights = {}
            for ticker, shares in holdings.items():
                if ticker in final_prices:
                    final_weights[ticker] = shares * final_prices[ticker] / final_portfolio_value
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(portfolio, benchmark)
            
            logger.info(f"Backtest completed from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"Final portfolio value: ${final_portfolio_value:.2f}")
            logger.info(f"Total return: {metrics['portfolio_return']:.2%}")
            
            return BacktestResult(
                portfolio_value_history=portfolio,
                benchmark_history=benchmark,
                trades=trades,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
                tickers=list(initial_weights.keys()),
                initial_weights=initial_weights,
                final_weights=final_weights
            )
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            raise
    
    def _calculate_metrics(self, portfolio: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            portfolio: Portfolio value history
            benchmark: Benchmark value history
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate returns
        portfolio_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
        benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
        
        # Calculate daily returns
        portfolio_daily_returns = portfolio.pct_change().dropna()
        benchmark_daily_returns = benchmark.pct_change().dropna()
        
        # Calculate annualized metrics
        days = len(portfolio)
        years = days / 252  # Approximate trading days in a year
        
        portfolio_annual_return = (1 + portfolio_return) ** (1 / years) - 1
        benchmark_annual_return = (1 + benchmark_return) ** (1 / years) - 1
        
        # Calculate volatility
        portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0% for simplicity)
        portfolio_sharpe = portfolio_annual_return / portfolio_volatility if portfolio_volatility > 0 else 0
        benchmark_sharpe = benchmark_annual_return / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # Calculate maximum drawdown
        portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
        portfolio_running_max = portfolio_cumulative.cummax()
        portfolio_drawdown = (portfolio_cumulative / portfolio_running_max) - 1
        portfolio_max_drawdown = portfolio_drawdown.min()
        
        benchmark_cumulative = (1 + benchmark_daily_returns).cumprod()
        benchmark_running_max = benchmark_cumulative.cummax()
        benchmark_drawdown = (benchmark_cumulative / benchmark_running_max) - 1
        benchmark_max_drawdown = benchmark_drawdown.min()
        
        # Calculate alpha and beta
        cov = np.cov(portfolio_daily_returns, benchmark_daily_returns)[0, 1]
        beta = cov / benchmark_daily_returns.var()
        alpha = portfolio_annual_return - (beta * benchmark_annual_return)
        
        return {
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "portfolio_annual_return": portfolio_annual_return,
            "benchmark_annual_return": benchmark_annual_return,
            "portfolio_volatility": portfolio_volatility,
            "benchmark_volatility": benchmark_volatility,
            "portfolio_sharpe": portfolio_sharpe,
            "benchmark_sharpe": benchmark_sharpe,
            "portfolio_max_drawdown": portfolio_max_drawdown,
            "benchmark_max_drawdown": benchmark_max_drawdown,
            "alpha": alpha,
            "beta": beta
        }
    
    def plot_backtest_results(self, result: BacktestResult, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results
        
        Args:
            result: BacktestResult object
            save_path: Path to save the plot
        """
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot portfolio and benchmark values
            ax1.plot(result.portfolio_value_history.index, result.portfolio_value_history, label='Portfolio', linewidth=2)
            ax1.plot(result.benchmark_history.index, result.benchmark_history, label='Benchmark', linewidth=2, alpha=0.7)
            
            # Add labels and title
            ax1.set_title('Portfolio Backtest Results', fontsize=16)
            ax1.set_ylabel('Value ($)', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True)
            
            # Annotate key metrics
            metrics_text = (
                f"Portfolio Return: {result.metrics['portfolio_return']:.2%}\n"
                f"Benchmark Return: {result.metrics['benchmark_return']:.2%}\n"
                f"Alpha: {result.metrics['alpha']:.2%}\n"
                f"Beta: {result.metrics['beta']:.2f}\n"
                f"Sharpe Ratio: {result.metrics['portfolio_sharpe']:.2f}\n"
                f"Max Drawdown: {result.metrics['portfolio_max_drawdown']:.2%}"
            )
            
            # Position the text box in figure coords
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Plot drawdown
            portfolio_daily_returns = result.portfolio_value_history.pct_change().dropna()
            portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
            portfolio_running_max = portfolio_cumulative.cummax()
            portfolio_drawdown = (portfolio_cumulative / portfolio_running_max) - 1
            
            ax2.fill_between(portfolio_drawdown.index, 0, portfolio_drawdown, color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True)
            
            # Format dates on x-axis
            fig.autofmt_xdate()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved backtest results plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting backtest results: {str(e)}")
            raise
    
    def save_trades_to_csv(self, result: BacktestResult, filepath: str) -> None:
        """
        Save trades to CSV file (FR-10)
        
        Args:
            result: BacktestResult object
            filepath: Path to save the CSV file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Write trades to CSV
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['date', 'ticker', 'action', 'shares', 'price', 'value', 'reason']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for trade in result.trades:
                    writer.writerow(trade)
            
            logger.info(f"Saved {len(result.trades)} trades to {filepath}")
        except Exception as e:
            logger.error(f"Error saving trades to CSV: {str(e)}")
            raise
