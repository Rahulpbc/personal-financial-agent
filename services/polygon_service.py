"""
Polygon API Service for fetching financial data
"""
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

class PolygonService:
    """Service for interacting with the Polygon.io API"""
    
    BASE_URL = "https://api.polygon.io"
    API_KEY = os.getenv("POLYGON_API_KEY")
    
    def __init__(self):
        if not self.API_KEY:
            raise ValueError("Polygon API key not found. Please set POLYGON_API_KEY in .env file.")
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the Polygon API"""
        if params is None:
            params = {}
        
        params["apiKey"] = self.API_KEY
        
        response = requests.get(f"{self.BASE_URL}{endpoint}", params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_current_price(self, ticker: str) -> Dict[str, Any]:
        """Get the current price for a ticker"""
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        return self._make_request(endpoint)
    
    def get_historical_prices(self, ticker: str, from_date: str, to_date: str, 
                             timespan: str = "day") -> Dict[str, Any]:
        """
        Get historical prices for a ticker
        
        Args:
            ticker: Stock ticker symbol
            from_date: Start date in format YYYY-MM-DD
            to_date: End date in format YYYY-MM-DD
            timespan: Time span (day, hour, minute, etc.)
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/{timespan}/{from_date}/{to_date}"
        return self._make_request(endpoint)
    
    def get_ticker_news(self, ticker: str, limit: int = 10) -> Dict[str, Any]:
        """Get latest news for a ticker"""
        endpoint = f"/v2/reference/news"
        params = {
            "ticker": ticker,
            "limit": limit,
            "order": "desc",
            "sort": "published_utc"
        }
        return self._make_request(endpoint, params)
    
    def get_ticker_details(self, ticker: str) -> Dict[str, Any]:
        """Get details for a ticker"""
        endpoint = f"/v3/reference/tickers/{ticker}"
        return self._make_request(endpoint)
    
    def get_financial_data(self, ticker: str, timeframe: str = "annual") -> Dict[str, Any]:
        """
        Get financial data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe for financial data (annual, quarterly)
        """
        endpoint = f"/vX/reference/financials/{ticker}"
        params = {
            "timeframe": timeframe,
            "limit": 5,  # Last 5 reports
            "sort": "period_of_report_date",
            "order": "desc",
            "include_sources": "true"
        }
        return self._make_request(endpoint, params)
