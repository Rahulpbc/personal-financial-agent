"""
Command Line Interface for the financial agent
"""
import click
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
import json
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database.db import get_db_session, create_tables
from data.database.models import User, Portfolio, Holding, StockPrice, NewsArticle, BacktestResult, TradeLog
from services.alpaca_service import AlpacaService
from services.polygon_service import PolygonService
from services.news_service import NewsService
from agents.advisor_agent import AdvisorAgent
from optimizer.portfolio_optimizer import PortfolioOptimizer
from backtesting.backtest import PortfolioBacktester
from utils.financial_metrics import extract_financial_data_from_polygon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.getcwd(), 'logs', f'financial_agent_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.getcwd(), 'logs'), exist_ok=True)

# Initialize services
try:
    alpaca_service = AlpacaService()
    logger.info("Alpaca service initialized")
except Exception as e:
    logger.warning(f"Failed to initialize Alpaca service: {str(e)}")
    alpaca_service = None

try:
    polygon_service = PolygonService()
    logger.info("Polygon service initialized")
except Exception as e:
    logger.warning(f"Failed to initialize Polygon service: {str(e)}")
    polygon_service = None

try:
    news_service = NewsService()
    logger.info("News service initialized")
except Exception as e:
    logger.warning(f"Failed to initialize News service: {str(e)}")
    news_service = None

# Create CLI group
@click.group()
def cli():
    """Financial Agent CLI"""
    # Create database tables if they don't exist
    create_tables()
    logger.info("Database tables created if they didn't exist")

@cli.command()
@click.option('--username', prompt='Username', help='Your username')
@click.option('--age', prompt='Age', type=int, help='Your age')
@click.option('--investment-horizon', prompt='Investment Horizon (years)', type=int, help='Your investment horizon in years')
@click.option('--risk-tolerance', prompt='Risk Tolerance', type=click.Choice(['low', 'medium', 'high']), help='Your risk tolerance')
@click.option('--target-return', prompt='Target Annual Return (%)', type=float, help='Your target annual return percentage')
def create_profile(username, age, investment_horizon, risk_tolerance, target_return):
    """Create a new user profile"""
    try:
        with get_db_session() as session:
            # Check if username already exists
            existing_user = session.query(User).filter(User.username == username).first()
            if existing_user:
                logger.error(f"Username {username} already exists")
                click.echo(f"Error: Username {username} already exists")
                return
            
            # Create new user
            user = User(
                username=username,
                age=age,
                investment_horizon=investment_horizon,
                risk_tolerance=risk_tolerance,
                target_return=target_return / 100.0  # Convert percentage to decimal
            )
            session.add(user)
            session.commit()
            
            logger.info(f"Created user profile for {username}")
            click.echo(f"Successfully created profile for {username}")
    except Exception as e:
        logger.error(f"Error creating user profile: {str(e)}")
        click.echo(f"Error: {str(e)}")

@cli.command()
@click.option('--username', prompt='Username', help='Your username')
@click.option('--name', prompt='Portfolio Name', help='Name of the portfolio')
@click.option('--description', prompt='Description', default='', help='Description of the portfolio')
def create_portfolio(username, name, description):
    """Create a new portfolio for a user"""
    try:
        with get_db_session() as session:
            # Get user
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.error(f"User {username} not found")
                click.echo(f"Error: User {username} not found")
                return
            
            # Create portfolio
            portfolio = Portfolio(
                user_id=user.id,
                name=name,
                description=description
            )
            session.add(portfolio)
            session.commit()
            
            logger.info(f"Created portfolio {name} for user {username}")
            click.echo(f"Successfully created portfolio {name} for user {username}")
    except Exception as e:
        logger.error(f"Error creating portfolio: {str(e)}")
        click.echo(f"Error: {str(e)}")

@cli.command()
@click.option('--username', prompt='Username', help='Your username')
@click.option('--portfolio-name', prompt='Portfolio Name', help='Name of the portfolio')
@click.option('--ticker', prompt='Ticker Symbol', help='Stock ticker symbol')
@click.option('--quantity', prompt='Quantity', type=float, help='Number of shares')
@click.option('--price', prompt='Purchase Price', type=float, help='Purchase price per share')
@click.option('--date', prompt='Purchase Date (YYYY-MM-DD)', default=datetime.now().strftime('%Y-%m-%d'), help='Purchase date')
def add_holding(username, portfolio_name, ticker, quantity, price, date):
    """Add a holding to a portfolio"""
    try:
        with get_db_session() as session:
            # Get user
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.error(f"User {username} not found")
                click.echo(f"Error: User {username} not found")
                return
            
            # Get portfolio
            portfolio = session.query(Portfolio).filter(
                Portfolio.user_id == user.id,
                Portfolio.name == portfolio_name
            ).first()
            
            if not portfolio:
                logger.error(f"Portfolio {portfolio_name} not found for user {username}")
                click.echo(f"Error: Portfolio {portfolio_name} not found for user {username}")
                return
            
            # Parse date
            purchase_date = datetime.strptime(date, '%Y-%m-%d')
            
            # Create holding
            holding = Holding(
                portfolio_id=portfolio.id,
                ticker=ticker.upper(),
                quantity=quantity,
                purchase_price=price,
                purchase_date=purchase_date
            )
            
            # Get current price
            try:
                if polygon_service:
                    current_price_data = polygon_service.get_current_price(ticker.upper())
                    if 'results' in current_price_data and current_price_data['results']:
                        holding.current_price = current_price_data['results'][0]['c']
                        holding.last_updated = datetime.now()
                elif alpaca_service:
                    current_price_data = alpaca_service.get_current_price(ticker.upper())
                    if 'price' in current_price_data:
                        holding.current_price = current_price_data['price']
                        holding.last_updated = datetime.now()
            except Exception as e:
                logger.warning(f"Could not get current price for {ticker}: {str(e)}")
            
            session.add(holding)
            session.commit()
            
            logger.info(f"Added {quantity} shares of {ticker} to portfolio {portfolio_name}")
            click.echo(f"Successfully added {quantity} shares of {ticker} to portfolio {portfolio_name}")
    except Exception as e:
        logger.error(f"Error adding holding: {str(e)}")
        click.echo(f"Error: {str(e)}")

@cli.command()
@click.option('--username', prompt='Username', help='Your username')
def list_portfolios(username):
    """List all portfolios for a user"""
    try:
        with get_db_session() as session:
            # Get user
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.error(f"User {username} not found")
                click.echo(f"Error: User {username} not found")
                return
            
            # Get portfolios
            portfolios = session.query(Portfolio).filter(Portfolio.user_id == user.id).all()
            
            if not portfolios:
                click.echo(f"No portfolios found for user {username}")
                return
            
            # Display portfolios
            click.echo(f"\nPortfolios for {username}:")
            click.echo("-" * 50)
            for portfolio in portfolios:
                click.echo(f"ID: {portfolio.id}, Name: {portfolio.name}")
                click.echo(f"Description: {portfolio.description}")
                click.echo(f"Created: {portfolio.created_at.strftime('%Y-%m-%d')}")
                
                # Get holdings
                holdings = session.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
                
                if holdings:
                    click.echo("\nHoldings:")
                    click.echo("-" * 50)
                    total_value = 0
                    for holding in holdings:
                        current_price = holding.current_price or holding.purchase_price
                        value = holding.quantity * current_price
                        total_value += value
                        
                        click.echo(f"Ticker: {holding.ticker}, Quantity: {holding.quantity}, " +
                                  f"Purchase Price: ${holding.purchase_price:.2f}, " +
                                  f"Current Price: ${current_price:.2f}, " +
                                  f"Value: ${value:.2f}")
                    
                    click.echo(f"\nTotal Portfolio Value: ${total_value:.2f}")
                else:
                    click.echo("No holdings in this portfolio")
                
                click.echo("\n" + "-" * 50)
    except Exception as e:
        logger.error(f"Error listing portfolios: {str(e)}")
        click.echo(f"Error: {str(e)}")

@cli.command()
@click.option('--username', prompt='Username', help='Your username')
@click.option('--portfolio-name', prompt='Portfolio Name', help='Name of the portfolio')
@click.option('--risk-tolerance', type=click.Choice(['low', 'medium', 'high']), help='Risk tolerance for optimization')
@click.option('--target-return', type=float, help='Target annual return percentage')
def optimize_portfolio(username, portfolio_name, risk_tolerance=None, target_return=None):
    """Optimize a portfolio using mean-variance optimization"""
    try:
        with get_db_session() as session:
            # Get user
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.error(f"User {username} not found")
                click.echo(f"Error: User {username} not found")
                return
            
            # Use user's risk tolerance and target return if not provided
            if risk_tolerance is None:
                risk_tolerance = user.risk_tolerance
            
            if target_return is None:
                target_return = user.target_return
            else:
                target_return = target_return / 100.0  # Convert percentage to decimal
            
            # Get portfolio
            portfolio = session.query(Portfolio).filter(
                Portfolio.user_id == user.id,
                Portfolio.name == portfolio_name
            ).first()
            
            if not portfolio:
                logger.error(f"Portfolio {portfolio_name} not found for user {username}")
                click.echo(f"Error: Portfolio {portfolio_name} not found for user {username}")
                return
            
            # Get holdings
            holdings = session.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
            
            if not holdings:
                logger.error(f"No holdings found in portfolio {portfolio_name}")
                click.echo(f"Error: No holdings found in portfolio {portfolio_name}")
                return
            
            # Get tickers
            tickers = [holding.ticker for holding in holdings]
            
            # Get historical prices for the last 1 year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            prices_data = {}
            
            for ticker in tickers:
                try:
                    if polygon_service:
                        historical_data = polygon_service.get_historical_prices(
                            ticker,
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                        
                        if 'results' in historical_data:
                            prices = [(datetime.fromtimestamp(result['t']/1000).date(), result['c']) 
                                     for result in historical_data['results']]
                            prices_data[ticker] = prices
                    elif alpaca_service:
                        historical_data = alpaca_service.get_historical_prices(
                            ticker,
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                        
                        if 'historical_data' in historical_data:
                            prices = [(pd.to_datetime(bar['timestamp']).date(), bar['close']) 
                                     for bar in historical_data['historical_data']]
                            prices_data[ticker] = prices
                except Exception as e:
                    logger.warning(f"Could not get historical prices for {ticker}: {str(e)}")
            
            if not prices_data:
                logger.error("Could not get historical prices for any ticker")
                click.echo("Error: Could not get historical prices for any ticker")
                return
            
            # Convert to DataFrame
            all_dates = sorted(set(date for ticker_prices in prices_data.values() for date, _ in ticker_prices))
            price_df = pd.DataFrame(index=all_dates, columns=prices_data.keys())
            
            for ticker, prices in prices_data.items():
                for date, price in prices:
                    price_df.loc[date, ticker] = price
            
            # Calculate current weights
            total_value = sum(holding.quantity * (holding.current_price or holding.purchase_price) for holding in holdings)
            current_weights = {holding.ticker: holding.quantity * (holding.current_price or holding.purchase_price) / total_value 
                             for holding in holdings}
            
            # Initialize optimizer
            optimizer = PortfolioOptimizer()
            
            # Optimize portfolio
            result = optimizer.optimize_portfolio(
                price_df,
                risk_tolerance=risk_tolerance,
                target_return=target_return,
                current_weights=current_weights
            )
            
            # Display results
            click.echo(f"\nPortfolio Optimization Results for {portfolio_name}:")
            click.echo("-" * 50)
            click.echo(f"Risk Tolerance: {risk_tolerance}")
            click.echo(f"Target Return: {target_return:.2%}")
            
            click.echo("\nCurrent Weights:")
            for ticker, weight in current_weights.items():
                click.echo(f"{ticker}: {weight:.2%}")
            
            click.echo("\nOptimized Weights:")
            for ticker, weight in result['weights'].items():
                click.echo(f"{ticker}: {weight:.2%}")
            
            click.echo("\nPerformance Metrics:")
            click.echo(f"Expected Annual Return: {result['performance']['expected_annual_return']:.2%}")
            click.echo(f"Annual Volatility: {result['performance']['annual_volatility']:.2%}")
            click.echo(f"Sharpe Ratio: {result['performance']['sharpe_ratio']:.2f}")
            
            click.echo("\nSuggested Allocation:")
            for ticker, shares in result['allocation'].items():
                click.echo(f"{ticker}: {shares} shares")
            
            click.echo(f"Leftover Cash: ${result['leftover_cash']:.2f}")
            
            # Generate and save chart
            try:
                chart_path = os.path.join(os.getcwd(), 'data', f'{username}_{portfolio_name}_optimization.png')
                optimizer.plot_portfolio_comparison(
                    current_weights,
                    result['weights'],
                    save_path=chart_path
                )
                click.echo(f"\nPortfolio comparison chart saved to: {chart_path}")
            except Exception as e:
                logger.warning(f"Could not generate portfolio comparison chart: {str(e)}")
            
            # Generate efficient frontier
            try:
                ef_data = optimizer.generate_efficient_frontier(price_df)
                ef_path = os.path.join(os.getcwd(), 'data', f'{username}_{portfolio_name}_efficient_frontier.png')
                
                # Calculate current portfolio performance
                current_portfolio = {
                    "return": ef_data['max_sharpe']['return'] * 0.8,  # Simplified example
                    "volatility": ef_data['max_sharpe']['volatility'] * 1.2  # Simplified example
                }
                
                optimizer.plot_efficient_frontier(
                    ef_data,
                    current_portfolio=current_portfolio,
                    save_path=ef_path
                )
                click.echo(f"Efficient frontier chart saved to: {ef_path}")
            except Exception as e:
                logger.warning(f"Could not generate efficient frontier chart: {str(e)}")
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {str(e)}")
        click.echo(f"Error: {str(e)}")

@cli.command()
@click.option('--username', prompt='Username', help='Your username')
@click.option('--portfolio-name', prompt='Portfolio Name', help='Name of the portfolio')
@click.option('--months', default=6, help='Number of months to backtest')
@click.option('--benchmark', default='SPY', help='Benchmark ticker symbol')
def backtest_portfolio(username, portfolio_name, months, benchmark):
    """Backtest a portfolio over the specified period"""
    try:
        with get_db_session() as session:
            # Get user
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.error(f"User {username} not found")
                click.echo(f"Error: User {username} not found")
                return
            
            # Get portfolio
            portfolio = session.query(Portfolio).filter(
                Portfolio.user_id == user.id,
                Portfolio.name == portfolio_name
            ).first()
            
            if not portfolio:
                logger.error(f"Portfolio {portfolio_name} not found for user {username}")
                click.echo(f"Error: Portfolio {portfolio_name} not found for user {username}")
                return
            
            # Get holdings
            holdings = session.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
            
            if not holdings:
                logger.error(f"No holdings found in portfolio {portfolio_name}")
                click.echo(f"Error: No holdings found in portfolio {portfolio_name}")
                return
            
            # Calculate weights
            total_value = sum(holding.quantity * (holding.current_price or holding.purchase_price) for holding in holdings)
            weights = {holding.ticker: holding.quantity * (holding.current_price or holding.purchase_price) / total_value 
                     for holding in holdings}
            
            # Get tickers including benchmark
            tickers = list(weights.keys()) + [benchmark]
            
            # Get historical prices
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30 * months)
            
            prices_data = {}
            
            for ticker in tickers:
                try:
                    if polygon_service:
                        historical_data = polygon_service.get_historical_prices(
                            ticker,
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                        
                        if 'results' in historical_data:
                            prices = [(datetime.fromtimestamp(result['t']/1000).date(), result['c']) 
                                     for result in historical_data['results']]
                            prices_data[ticker] = prices
                    elif alpaca_service:
                        historical_data = alpaca_service.get_historical_prices(
                            ticker,
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                        
                        if 'historical_data' in historical_data:
                            prices = [(pd.to_datetime(bar['timestamp']).date(), bar['close']) 
                                     for bar in historical_data['historical_data']]
                            prices_data[ticker] = prices
                except Exception as e:
                    logger.warning(f"Could not get historical prices for {ticker}: {str(e)}")
            
            if not prices_data:
                logger.error("Could not get historical prices for any ticker")
                click.echo("Error: Could not get historical prices for any ticker")
                return
            
            # Convert to DataFrame
            all_dates = sorted(set(date for ticker_prices in prices_data.values() for date, _ in ticker_prices))
            price_df = pd.DataFrame(index=all_dates, columns=prices_data.keys())
            
            for ticker, prices in prices_data.items():
                for date, price in prices:
                    price_df.loc[date, ticker] = price
            
            # Initialize backtester
            backtester = PortfolioBacktester()
            
            # Run backtest
            result = backtester.backtest_portfolio(
                price_df,
                initial_weights=weights,
                benchmark_ticker=benchmark,
                rebalance_frequency='M',  # Monthly rebalancing
                initial_capital=10000.0
            )
            
            # Display results
            click.echo(f"\nBacktest Results for {portfolio_name}:")
            click.echo("-" * 50)
            click.echo(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
            click.echo(f"Initial Portfolio Value: ${10000:.2f}")
            click.echo(f"Final Portfolio Value: ${result.portfolio_value_history.iloc[-1]:.2f}")
            
            click.echo("\nPerformance Metrics:")
            click.echo(f"Portfolio Return: {result.metrics['portfolio_return']:.2%}")
            click.echo(f"Benchmark Return: {result.metrics['benchmark_return']:.2%}")
            click.echo(f"Alpha: {result.metrics['alpha']:.2%}")
            click.echo(f"Beta: {result.metrics['beta']:.2f}")
            click.echo(f"Sharpe Ratio: {result.metrics['portfolio_sharpe']:.2f}")
            click.echo(f"Maximum Drawdown: {result.metrics['portfolio_max_drawdown']:.2%}")
            
            # Generate and save chart
            try:
                chart_path = os.path.join(os.getcwd(), 'data', f'{username}_{portfolio_name}_backtest.png')
                backtester.plot_backtest_results(result, save_path=chart_path)
                click.echo(f"\nBacktest chart saved to: {chart_path}")
            except Exception as e:
                logger.warning(f"Could not generate backtest chart: {str(e)}")
            
            # Save trades to CSV
            try:
                trades_path = os.path.join(os.getcwd(), 'data', f'{username}_{portfolio_name}_trades.csv')
                backtester.save_trades_to_csv(result, trades_path)
                click.echo(f"Trade log saved to: {trades_path}")
            except Exception as e:
                logger.warning(f"Could not save trade log: {str(e)}")
            
            # Save backtest result to database
            try:
                backtest_result = BacktestResult(
                    user_id=user.id,
                    portfolio_id=portfolio.id,
                    start_date=result.start_date,
                    end_date=result.end_date,
                    initial_value=10000.0,
                    final_value=result.portfolio_value_history.iloc[-1],
                    return_pct=result.metrics['portfolio_return'],
                    benchmark_return_pct=result.metrics['benchmark_return'],
                    sharpe_ratio=result.metrics['portfolio_sharpe'],
                    max_drawdown=result.metrics['portfolio_max_drawdown']
                )
                session.add(backtest_result)
                
                # Save trades to database
                for trade in result.trades:
                    trade_log = TradeLog(
                        portfolio_id=portfolio.id,
                        ticker=trade['ticker'],
                        trade_date=datetime.strptime(trade['date'], '%Y-%m-%d'),
                        action=trade['action'],
                        quantity=trade['shares'],
                        price=trade['price'],
                        total_value=trade['value'],
                        reason=trade['reason']
                    )
                    session.add(trade_log)
                
                session.commit()
                logger.info(f"Saved backtest results for portfolio {portfolio_name}")
            except Exception as e:
                logger.warning(f"Could not save backtest results to database: {str(e)}")
    except Exception as e:
        logger.error(f"Error backtesting portfolio: {str(e)}")
        click.echo(f"Error: {str(e)}")

@cli.command()
@click.option('--username', prompt='Username', help='Your username')
@click.option('--query', prompt='Question', help='Your question for the advisor')
async def ask_advisor(username, query):
    """Ask the financial advisor a question"""
    try:
        with get_db_session() as session:
            # Get user
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.error(f"User {username} not found")
                click.echo(f"Error: User {username} not found")
                return
            
            # Initialize advisor agent
            advisor = AdvisorAgent(use_polygon=bool(polygon_service))
            
            # Generate advice
            advice = await advisor.generate_advice(user, query)
            
            # Display advice
            click.echo(f"\nFinancial Advice for {username}:")
            click.echo("-" * 50)
            click.echo(advice['advice'])
            
            logger.info(f"Generated advice for user {username}")
    except Exception as e:
        logger.error(f"Error generating advice: {str(e)}")
        click.echo(f"Error: {str(e)}")

if __name__ == '__main__':
    cli()
