"""
Configuration settings for the financial agent
"""
import os
import logging
from dotenv import load_dotenv
from pathlib import Path
import random
import numpy as np

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/financial_agent.db")

# API Keys (NFR-8: Don't hard-code credentials)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Random seed for reproducibility (NFR-3)
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / f"financial_agent_{os.getenv('LOG_FILE_SUFFIX', '')}.log"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# RAG settings
MAX_NEWS_ARTICLES = int(os.getenv("MAX_NEWS_ARTICLES", "10000"))  # NFR-5: Limit dataset size

# Portfolio optimization settings
DEFAULT_RISK_FREE_RATE = float(os.getenv("DEFAULT_RISK_FREE_RATE", "0.02"))

# Backtesting settings
DEFAULT_BACKTEST_MONTHS = int(os.getenv("DEFAULT_BACKTEST_MONTHS", "6"))  # FR-9: 6 months
DEFAULT_BENCHMARK = os.getenv("DEFAULT_BENCHMARK", "SPY")

# API rate limiting
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "5"))  # Requests per second

# Performance settings
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "5"))  # NFR-4: Under 5 seconds

# Default tickers for demo
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "TSLA", "NVDA", "BRK.B", "JPM", "JNJ"
]
