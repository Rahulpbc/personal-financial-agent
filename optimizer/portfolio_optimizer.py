"""
Portfolio optimization module using PyPortfolioOpt
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions, DiscreteAllocation

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Portfolio optimization using PyPortfolioOpt"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the portfolio optimizer
        
        Args:
            random_seed: Random seed for reproducibility (NFR-3)
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        logger.info(f"Portfolio optimizer initialized with random seed {random_seed}")
    
    def _prepare_data(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare price data for optimization
        
        Args:
            prices_df: DataFrame with price data (index=dates, columns=tickers)
            
        Returns:
            Cleaned DataFrame
        """
        # Check if we have enough data
        if prices_df.shape[0] < 30:
            logger.warning(f"Limited price data available: {prices_df.shape[0]} days")
        
        # Forward fill missing values
        prices_df = prices_df.ffill()
        
        # Drop tickers with too many missing values
        missing_threshold = 0.3  # 30% missing values
        missing_pct = prices_df.isna().mean()
        tickers_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        
        if tickers_to_drop:
            logger.warning(f"Dropping tickers with >30% missing data: {tickers_to_drop}")
            prices_df = prices_df.drop(columns=tickers_to_drop)
        
        # Fill any remaining NaN values with 0
        prices_df = prices_df.fillna(0)
        
        return prices_df
    
    def optimize_portfolio(
        self,
        prices_df: pd.DataFrame,
        risk_tolerance: str = "medium",
        target_return: Optional[float] = None,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights
        
        Args:
            prices_df: DataFrame with price data (index=dates, columns=tickers)
            risk_tolerance: Risk tolerance level (low, medium, high)
            target_return: Target annual return (as decimal)
            current_weights: Current portfolio weights (ticker -> weight)
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Prepare data
            prices_df = self._prepare_data(prices_df)
            
            # Calculate expected returns and covariance matrix
            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)
            
            # Initialize efficient frontier
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            
            # Add regularization for stability
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            
            # Optimize based on risk tolerance
            if risk_tolerance == "low":
                logger.info("Optimizing for minimum volatility")
                ef.min_volatility()
            elif risk_tolerance == "high":
                logger.info("Optimizing for maximum Sharpe ratio")
                ef.max_sharpe()
            elif target_return is not None:
                logger.info(f"Optimizing for target return: {target_return}")
                ef.efficient_return(target_return=target_return)
            else:
                logger.info("Optimizing for maximum quadratic utility")
                ef.max_quadratic_utility()
            
            # Get optimized weights
            weights = ef.clean_weights()
            
            # Calculate performance metrics
            expected_annual_return = ef.portfolio_performance()[0]
            annual_volatility = ef.portfolio_performance()[1]
            sharpe_ratio = ef.portfolio_performance()[2]
            
            # Calculate allocation
            latest_prices = prices_df.iloc[-1]
            portfolio_value = 10000  # Default value for calculation
            if current_weights:
                # Calculate current portfolio value
                current_value = sum(latest_prices[ticker] * current_weights.get(ticker, 0) for ticker in current_weights)
                if current_value > 0:
                    portfolio_value = current_value
            
            # Get discrete allocation
            da = DiscreteAllocation(weights, latest_prices, portfolio_value)
            allocation, leftover = da.greedy_portfolio()
            
            logger.info(f"Portfolio optimization completed with expected return: {expected_annual_return:.4f}")
            
            return {
                "weights": weights,
                "allocation": allocation,
                "leftover_cash": leftover,
                "performance": {
                    "expected_annual_return": expected_annual_return,
                    "annual_volatility": annual_volatility,
                    "sharpe_ratio": sharpe_ratio
                },
                "tickers": list(weights.keys()),
                "portfolio_value": portfolio_value
            }
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            raise
    
    def generate_efficient_frontier(
        self,
        prices_df: pd.DataFrame,
        n_points: int = 20,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Generate efficient frontier
        
        Args:
            prices_df: DataFrame with price data (index=dates, columns=tickers)
            n_points: Number of points on the efficient frontier
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary with efficient frontier data
        """
        try:
            # Prepare data
            prices_df = self._prepare_data(prices_df)
            
            # Calculate expected returns and covariance matrix
            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)
            
            # Calculate efficient frontier
            returns = []
            volatilities = []
            sharpe_ratios = []
            
            # Generate range of target returns
            min_return = mu.min()
            max_return = mu.max()
            target_returns = np.linspace(min_return, max_return, n_points)
            
            for target_return in target_returns:
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                try:
                    ef.efficient_return(target_return=target_return)
                    returns.append(ef.portfolio_performance()[0])
                    volatilities.append(ef.portfolio_performance()[1])
                    sharpe_ratios.append(ef.portfolio_performance()[2])
                except:
                    # Skip infeasible target returns
                    continue
            
            # Calculate maximum Sharpe ratio portfolio
            ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
            max_sharpe_return = ef_max_sharpe.portfolio_performance()[0]
            max_sharpe_volatility = ef_max_sharpe.portfolio_performance()[1]
            max_sharpe_ratio = ef_max_sharpe.portfolio_performance()[2]
            
            # Calculate minimum volatility portfolio
            ef_min_vol = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef_min_vol.min_volatility()
            min_vol_return = ef_min_vol.portfolio_performance()[0]
            min_vol_volatility = ef_min_vol.portfolio_performance()[1]
            min_vol_sharpe = ef_min_vol.portfolio_performance()[2]
            
            logger.info("Generated efficient frontier")
            
            return {
                "efficient_frontier": {
                    "returns": returns,
                    "volatilities": volatilities,
                    "sharpe_ratios": sharpe_ratios
                },
                "max_sharpe": {
                    "return": max_sharpe_return,
                    "volatility": max_sharpe_volatility,
                    "sharpe_ratio": max_sharpe_ratio,
                    "weights": ef_max_sharpe.clean_weights()
                },
                "min_volatility": {
                    "return": min_vol_return,
                    "volatility": min_vol_volatility,
                    "sharpe_ratio": min_vol_sharpe,
                    "weights": ef_min_vol.clean_weights()
                }
            }
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {str(e)}")
            raise
    
    def plot_efficient_frontier(
        self,
        ef_data: Dict[str, Any],
        current_portfolio: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot efficient frontier
        
        Args:
            ef_data: Efficient frontier data from generate_efficient_frontier
            current_portfolio: Current portfolio (return, volatility)
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot efficient frontier
            plt.plot(
                ef_data["efficient_frontier"]["volatilities"],
                ef_data["efficient_frontier"]["returns"],
                'b-', linewidth=2, label="Efficient Frontier"
            )
            
            # Plot maximum Sharpe ratio portfolio
            plt.scatter(
                ef_data["max_sharpe"]["volatility"],
                ef_data["max_sharpe"]["return"],
                marker='*', color='r', s=150, label=f"Maximum Sharpe Ratio: {ef_data['max_sharpe']['sharpe_ratio']:.2f}"
            )
            
            # Plot minimum volatility portfolio
            plt.scatter(
                ef_data["min_volatility"]["volatility"],
                ef_data["min_volatility"]["return"],
                marker='o', color='g', s=150, label=f"Minimum Volatility: {ef_data['min_volatility']['return']:.2f}"
            )
            
            # Plot current portfolio if provided
            if current_portfolio:
                plt.scatter(
                    current_portfolio["volatility"],
                    current_portfolio["return"],
                    marker='D', color='purple', s=150, label="Current Portfolio"
                )
            
            plt.title('Efficient Frontier', fontsize=16)
            plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
            plt.ylabel('Expected Return', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved efficient frontier plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting efficient frontier: {str(e)}")
            raise
    
    def plot_portfolio_comparison(
        self,
        current_weights: Dict[str, float],
        optimized_weights: Dict[str, float],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison between current and optimized portfolio
        
        Args:
            current_weights: Current portfolio weights
            optimized_weights: Optimized portfolio weights
            save_path: Path to save the plot
        """
        try:
            # Get all tickers
            all_tickers = sorted(set(list(current_weights.keys()) + list(optimized_weights.keys())))
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'Current': [current_weights.get(ticker, 0) for ticker in all_tickers],
                'Optimized': [optimized_weights.get(ticker, 0) for ticker in all_tickers]
            }, index=all_tickers)
            
            # Sort by optimized weights
            df = df.sort_values('Optimized', ascending=False)
            
            # Plot
            plt.figure(figsize=(12, 8))
            df.plot(kind='bar', width=0.8)
            
            plt.title('Portfolio Allocation Comparison', fontsize=16)
            plt.xlabel('Ticker', fontsize=12)
            plt.ylabel('Weight', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            
            # Add text labels
            for i, ticker in enumerate(df.index):
                current = df.loc[ticker, 'Current']
                optimized = df.loc[ticker, 'Optimized']
                change = optimized - current
                
                if abs(change) > 0.01:  # Only show significant changes
                    color = 'green' if change > 0 else 'red'
                    plt.text(
                        i, 
                        max(current, optimized) + 0.02, 
                        f"{change:+.1%}", 
                        ha='center', 
                        color=color,
                        fontweight='bold'
                    )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved portfolio comparison plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting portfolio comparison: {str(e)}")
            raise
