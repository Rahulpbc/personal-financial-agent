"""
Alpaca API Service for fetching financial data
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import alpaca_trade_api as tradeapi
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AlpacaService:
    """Service for interacting with the Alpaca API"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_API_SECRET")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        if not self.api_key or not self.api_secret:
            logger.error("Alpaca API credentials not found")
            raise ValueError("Alpaca API key and secret not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET in .env file.")
        
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
        logger.info("Alpaca API client initialized")
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.api.get_account()
            logger.info("Retrieved account information")
            return {
                "id": account.id,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "status": account.status
            }
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            raise
    
    def get_current_price(self, ticker: str) -> Dict[str, Any]:
        """Get the current price for a ticker"""
        try:
            # Get the latest bar
            barset = self.api.get_latest_bar(ticker)
            logger.info(f"Retrieved current price for {ticker}")
            
            return {
                "ticker": ticker,
                "price": float(barset.c),
                "timestamp": barset.t.isoformat(),
                "open": float(barset.o),
                "high": float(barset.h),
                "low": float(barset.l),
                "volume": int(barset.v)
            }
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
            raise
    
    def get_historical_prices(
        self, 
        ticker: str, 
        from_date: str, 
        to_date: str, 
        timeframe: str = "1Day"
    ) -> Dict[str, Any]:
        """
        Get historical prices for a ticker
        
        Args:
            ticker: Stock ticker symbol
            from_date: Start date in format YYYY-MM-DD
            to_date: End date in format YYYY-MM-DD
            timeframe: Time frame (1Day, 1Hour, 1Min, etc.)
        """
        try:
            # Convert string dates to datetime objects
            start_dt = pd.Timestamp(from_date).tz_localize('UTC')
            end_dt = pd.Timestamp(to_date).tz_localize('UTC')
            
            # Get historical bars
            bars = self.api.get_bars(ticker, timeframe, start_dt, end_dt).df
            
            # Reset index to make the timestamp a column
            bars = bars.reset_index()
            
            # Convert to dictionary
            bars_dict = bars.to_dict(orient='records')
            
            logger.info(f"Retrieved historical prices for {ticker} from {from_date} to {to_date}")
            
            return {
                "ticker": ticker,
                "historical_data": bars_dict,
                "parameters": {
                    "from_date": from_date,
                    "to_date": to_date,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            logger.error(f"Error getting historical prices for {ticker}: {str(e)}")
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            logger.info("Retrieved current positions")
            
            result = []
            for position in positions:
                result.append({
                    "ticker": position.symbol,
                    "quantity": float(position.qty),
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "current_price": float(position.current_price),
                    "lastday_price": float(position.lastday_price),
                    "change_today": float(position.change_today)
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise
    
    def get_portfolio_history(
        self, 
        period: str = "1M", 
        timeframe: str = "1D"
    ) -> Dict[str, Any]:
        """
        Get portfolio history
        
        Args:
            period: Time period (1D, 1M, 3M, 1A, etc.)
            timeframe: Time frame for data points (1D, 1H, etc.)
        """
        try:
            history = self.api.get_portfolio_history(period=period, timeframe=timeframe)
            logger.info(f"Retrieved portfolio history for period {period}")
            
            return {
                "timestamp": [ts.isoformat() for ts in history.timestamp],
                "equity": history.equity,
                "profit_loss": history.profit_loss,
                "profit_loss_pct": history.profit_loss_pct,
                "base_value": history.base_value,
                "timeframe": timeframe
            }
        except Exception as e:
            logger.error(f"Error getting portfolio history: {str(e)}")
            raise
    
    def execute_order(
        self, 
        ticker: str, 
        qty: float, 
        side: str, 
        order_type: str = "market", 
        time_in_force: str = "day"
    ) -> Dict[str, Any]:
        """
        Execute an order
        
        Args:
            ticker: Stock ticker symbol
            qty: Quantity to buy/sell
            side: buy or sell
            order_type: market, limit, stop, stop_limit
            time_in_force: day, gtc, opg, cls, ioc, fok
        """
        try:
            order = self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            
            logger.info(f"Executed {side} order for {qty} shares of {ticker}")
            
            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "ticker": order.symbol,
                "quantity": float(order.qty),
                "side": order.side,
                "order_type": order.type,
                "status": order.status,
                "created_at": order.created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Error executing order for {ticker}: {str(e)}")
            raise
    
    def get_market_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get market calendar
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
        """
        try:
            calendar = self.api.get_calendar(start=start_date, end=end_date)
            logger.info(f"Retrieved market calendar from {start_date} to {end_date}")
            
            result = []
            for day in calendar:
                result.append({
                    "date": day.date.isoformat(),
                    "open": day.open.isoformat(),
                    "close": day.close.isoformat(),
                    "session_open": day.session_open.isoformat(),
                    "session_close": day.session_close.isoformat()
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting market calendar: {str(e)}")
            raise
