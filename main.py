"""
Main entry point for the financial agent application
"""
import os
import sys
import click
import logging
import asyncio
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config.settings import (
    LOGS_DIR, LOG_LEVEL, LOG_FORMAT, 
    POLYGON_API_KEY, ALPACA_API_KEY, NEWS_API_KEY, OPENAI_API_KEY,
    DEFAULT_TICKERS
)

# Import database utilities
from data.database.db import create_tables
from data.database.models import User, Portfolio, Holding

# Import services
from services.polygon_service import PolygonService
from services.alpaca_service import AlpacaService
from services.news_service import NewsService

# Import agent
from agents.advisor_agent import AdvisorAgent

# Import UI
from ui.cli import cli as cli_app

# Import utilities
from utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger(
    log_file=os.path.join(LOGS_DIR, f"financial_agent_{datetime.now().strftime('%Y%m%d')}.log"),
    log_level=getattr(logging, LOG_LEVEL)
)

@click.group()
def cli():
    """Personal Financial Agent CLI"""
    pass

@cli.command()
def setup():
    """Set up the financial agent application"""
    try:
        logger.info("Setting up financial agent application")
        
        # Create database tables
        create_tables()
        logger.info("Database tables created")
        
        # Check API keys
        missing_keys = []
        if not POLYGON_API_KEY:
            missing_keys.append("POLYGON_API_KEY")
        if not ALPACA_API_KEY:
            missing_keys.append("ALPACA_API_KEY")
        if not NEWS_API_KEY:
            missing_keys.append("NEWS_API_KEY")
        if not OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
            click.echo(f"Warning: The following API keys are missing: {', '.join(missing_keys)}")
            click.echo("Please add them to your .env file to enable all features.")
        
        # Initialize services to test connections
        try:
            if POLYGON_API_KEY:
                polygon_service = PolygonService()
                logger.info("Polygon API connection successful")
                click.echo("Polygon API connection successful")
        except Exception as e:
            logger.error(f"Polygon API connection failed: {str(e)}")
            click.echo(f"Error: Polygon API connection failed: {str(e)}")
        
        try:
            if ALPACA_API_KEY:
                alpaca_service = AlpacaService()
                logger.info("Alpaca API connection successful")
                click.echo("Alpaca API connection successful")
        except Exception as e:
            logger.error(f"Alpaca API connection failed: {str(e)}")
            click.echo(f"Error: Alpaca API connection failed: {str(e)}")
        
        try:
            if NEWS_API_KEY:
                news_service = NewsService()
                logger.info("News API connection successful")
                click.echo("News API connection successful")
        except Exception as e:
            logger.error(f"News API connection failed: {str(e)}")
            click.echo(f"Error: News API connection failed: {str(e)}")
        
        click.echo("Setup completed successfully")
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        click.echo(f"Error: Setup failed: {str(e)}")

@cli.command()
@click.option('--tickers', default=','.join(DEFAULT_TICKERS), help='Comma-separated list of tickers to fetch data for')
def fetch_initial_data(tickers):
    """Fetch initial data for the financial agent"""
    try:
        logger.info("Fetching initial data")
        
        # Parse tickers
        ticker_list = [ticker.strip() for ticker in tickers.split(',')]
        
        # Fetch news data
        if NEWS_API_KEY:
            click.echo("Fetching and indexing news data...")
            news_service = NewsService()
            news_service.fetch_and_index_for_tickers(ticker_list)
            click.echo("News data fetched and indexed successfully")
        
        # Fetch financial data
        if POLYGON_API_KEY:
            click.echo("Fetching financial data from Polygon API...")
            polygon_service = PolygonService()
            
            for ticker in ticker_list:
                try:
                    # Get current price
                    current_price = polygon_service.get_current_price(ticker)
                    click.echo(f"Current price for {ticker}: ${current_price['results'][0]['c']:.2f}")
                    
                    # Get financial data
                    financial_data = polygon_service.get_financial_data(ticker)
                    click.echo(f"Financial data fetched for {ticker}")
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {str(e)}")
                    click.echo(f"Error fetching data for {ticker}: {str(e)}")
        
        click.echo("Initial data fetched successfully")
    except Exception as e:
        logger.error(f"Error fetching initial data: {str(e)}")
        click.echo(f"Error: {str(e)}")

@cli.command()
def run_cli():
    """Run the CLI application"""
    cli_app()

@cli.command()
def run_api():
    """Run the FastAPI application"""
    try:
        import uvicorn
        from app.main import app
        
        logger.info("Starting FastAPI application")
        click.echo("Starting FastAPI application at http://localhost:8000")
        click.echo("API documentation available at http://localhost:8000/docs")
        
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Error starting FastAPI application: {str(e)}")
        click.echo(f"Error: {str(e)}")

if __name__ == "__main__":
    cli()
