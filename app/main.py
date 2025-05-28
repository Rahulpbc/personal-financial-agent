"""
FastAPI application for the financial agent
"""
import os
import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json

# Import configuration
from config.settings import (
    LOGS_DIR, LOG_LEVEL, LOG_FORMAT, 
    POLYGON_API_KEY, ALPACA_API_KEY, NEWS_API_KEY, OPENAI_API_KEY,
    DEFAULT_TICKERS, DATA_DIR
)

# Import database models and utilities
from data.database.db import get_db_session, create_tables
from data.database.models import User, Portfolio, Holding, StockPrice, NewsArticle, BacktestResult, TradeLog

# Import data models
from data.models import (
    TickerRequest, 
    HistoricalDataRequest, 
    NewsRequest, 
    FinancialDataRequest, 
    DCFValuationRequest, 
    FinancialMetricsResponse, 
    DCFValuationResponse,
    AgentQueryRequest,
    # New models for enhanced requirements
    UserProfileCreate,
    UserProfileResponse,
    PortfolioCreate,
    PortfolioResponse,
    HoldingCreate,
    HoldingResponse,
    PortfolioOptimizationRequest,
    PortfolioOptimizationResponse,
    BacktestRequest,
    BacktestResponse,
    AdvisorQueryRequest,
    AdvisorQueryResponse
)

# Import services
from services.polygon_service import PolygonService
from services.alpaca_service import AlpacaService
from services.news_service import NewsService

# Import agents
from models.financial_agent import FinancialAgent
from agents.advisor_agent import AdvisorAgent

# Import optimization and backtesting
from optimizer.portfolio_optimizer import PortfolioOptimizer
from backtesting.backtest import PortfolioBacktester

# Import utilities
from utils.financial_metrics import extract_financial_data_from_polygon
from utils.logging_utils import setup_logger, log_api_call, handle_api_error

# Configure logging
logger = setup_logger(
    log_file=os.path.join(LOGS_DIR, f"api_{datetime.now().strftime('%Y%m%d')}.log"),
    log_level=getattr(logging, LOG_LEVEL)
)

# Create database tables
create_tables()

# Create FastAPI app
app = FastAPI(
    title="Personal Financial Agent",
    description="A financial analysis agent that leverages financial APIs to access data, perform calculations, and provide investment advice.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
try:
    polygon_service = PolygonService() if POLYGON_API_KEY else None
    logger.info("Polygon service initialized" if polygon_service else "Polygon service not available - missing API key")
except Exception as e:
    logger.error(f"Failed to initialize Polygon service: {str(e)}")
    polygon_service = None

try:
    alpaca_service = AlpacaService() if ALPACA_API_KEY else None
    logger.info("Alpaca service initialized" if alpaca_service else "Alpaca service not available - missing API key")
except Exception as e:
    logger.error(f"Failed to initialize Alpaca service: {str(e)}")
    alpaca_service = None

try:
    news_service = NewsService() if NEWS_API_KEY else None
    logger.info("News service initialized" if news_service else "News service not available - missing API key")
except Exception as e:
    logger.error(f"Failed to initialize News service: {str(e)}")
    news_service = None

# Initialize agents and tools
financial_agent = FinancialAgent()
advisor_agent = AdvisorAgent(use_polygon=bool(polygon_service))
portfolio_optimizer = PortfolioOptimizer()
backtester = PortfolioBacktester()

# Create data directories if they don't exist
os.makedirs(os.path.join(DATA_DIR), exist_ok=True)

# Database dependency
def get_db():
    """Get database session"""
    with get_db_session() as session:
        yield session

# Helper functions
def get_data_service():
    """Get available data service (Polygon or Alpaca)"""
    if polygon_service:
        return polygon_service
    elif alpaca_service:
        return alpaca_service
    else:
        raise HTTPException(
            status_code=503,
            detail="No financial data service available. Please configure Polygon or Alpaca API keys."
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Personal Financial Agent API",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.post("/api/price/current")
async def get_current_price(request: TickerRequest):
    """Get current price for a ticker"""
    try:
        result = polygon_service.get_current_price(request.ticker)
        return {
            "ticker": request.ticker,
            "price_data": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/price/historical")
async def get_historical_prices(request: HistoricalDataRequest):
    """Get historical prices for a ticker"""
    try:
        result = polygon_service.get_historical_prices(
            request.ticker, 
            request.from_date, 
            request.to_date, 
            request.timespan
        )
        return {
            "ticker": request.ticker,
            "historical_data": result,
            "parameters": {
                "from_date": request.from_date,
                "to_date": request.to_date,
                "timespan": request.timespan
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/news")
async def get_ticker_news(request: NewsRequest):
    """Get latest news for a ticker"""
    try:
        result = polygon_service.get_ticker_news(request.ticker, request.limit)
        return {
            "ticker": request.ticker,
            "news_data": result,
            "parameters": {
                "limit": request.limit
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/financials")
async def get_financial_data(request: FinancialDataRequest):
    """Get financial data for a ticker"""
    try:
        result = polygon_service.get_financial_data(request.ticker, request.timeframe)
        return {
            "ticker": request.ticker,
            "financial_data": result,
            "parameters": {
                "timeframe": request.timeframe
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metrics", response_model=FinancialMetricsResponse)
async def get_financial_metrics(request: FinancialDataRequest):
    """Get financial metrics for a ticker including owner earnings, ROE, and ROIC"""
    try:
        result = await financial_agent.get_financial_metrics(request.ticker, request.timeframe)
        return {
            "ticker": request.ticker,
            "metrics": result.get("metrics", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/valuation/dcf", response_model=DCFValuationResponse)
async def perform_dcf_valuation(request: DCFValuationRequest):
    """Perform a discounted cash flow (DCF) valuation for a ticker"""
    try:
        result = await financial_agent.perform_dcf_valuation(
            request.ticker,
            request.projected_growth_rates,
            request.terminal_growth_rate,
            request.discount_rate,
            request.projection_years
        )
        return {
            "ticker": request.ticker,
            "valuation": result.get("valuation", {}),
            "assumptions": result.get("assumptions", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/query")
async def process_agent_query(request: AgentQueryRequest):
    """Process a natural language query about a ticker"""
    try:
        result = await financial_agent.process_query(request.ticker, request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# User Profile Endpoints
@app.post("/api/users", response_model=UserProfileResponse, tags=["User Profiles"])
@log_api_call
async def create_user_profile(user: UserProfileCreate, db: Session = Depends(get_db)):
    """Create a new user profile (FR-1)"""
    try:
        # Check if username already exists
        existing_user = db.query(User).filter(User.username == user.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail=f"Username {user.username} already exists")
        
        # Create new user
        new_user = User(
            username=user.username,
            age=user.age,
            investment_horizon=user.investment_horizon,
            risk_tolerance=user.risk_tolerance.value,
            target_return=user.target_return
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"Created user profile for {user.username}")
        return new_user
    except Exception as e:
        logger.error(f"Error creating user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{username}", response_model=UserProfileResponse, tags=["User Profiles"])
@log_api_call
async def get_user_profile(username: str, db: Session = Depends(get_db)):
    """Get a user profile by username"""
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User {username} not found")
        
        return user
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Portfolio Management Endpoints
@app.post("/api/users/{username}/portfolios", response_model=PortfolioResponse, tags=["Portfolios"])
@log_api_call
async def create_portfolio(username: str, portfolio: PortfolioCreate, db: Session = Depends(get_db)):
    """Create a new portfolio for a user (FR-2)"""
    try:
        # Get user
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User {username} not found")
        
        # Create portfolio
        new_portfolio = Portfolio(
            user_id=user.id,
            name=portfolio.name,
            description=portfolio.description
        )
        db.add(new_portfolio)
        db.commit()
        db.refresh(new_portfolio)
        
        logger.info(f"Created portfolio {portfolio.name} for user {username}")
        return new_portfolio
    except Exception as e:
        logger.error(f"Error creating portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{username}/portfolios", response_model=List[PortfolioResponse], tags=["Portfolios"])
@log_api_call
async def get_user_portfolios(username: str, db: Session = Depends(get_db)):
    """Get all portfolios for a user"""
    try:
        # Get user
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User {username} not found")
        
        # Get portfolios
        portfolios = db.query(Portfolio).filter(Portfolio.user_id == user.id).all()
        
        # Calculate total value and holdings count for each portfolio
        result = []
        for portfolio in portfolios:
            holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
            total_value = 0
            for holding in holdings:
                current_price = holding.current_price or holding.purchase_price
                total_value += holding.quantity * current_price
            
            portfolio_dict = portfolio.__dict__.copy()
            portfolio_dict["total_value"] = total_value
            portfolio_dict["holdings_count"] = len(holdings)
            result.append(portfolio_dict)
        
        return result
    except Exception as e:
        logger.error(f"Error getting portfolios: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolios/{portfolio_id}/holdings", response_model=HoldingResponse, tags=["Holdings"])
@log_api_call
async def add_holding(portfolio_id: int, holding: HoldingCreate, db: Session = Depends(get_db)):
    """Add a holding to a portfolio"""
    try:
        # Get portfolio
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail=f"Portfolio with ID {portfolio_id} not found")
        
        # Create holding
        new_holding = Holding(
            portfolio_id=portfolio_id,
            ticker=holding.ticker.upper(),
            quantity=holding.quantity,
            purchase_price=holding.purchase_price,
            purchase_date=holding.purchase_date
        )
        
        # Get current price
        try:
            data_service = get_data_service()
            if isinstance(data_service, PolygonService):
                current_price_data = data_service.get_current_price(holding.ticker.upper())
                if 'results' in current_price_data and current_price_data['results']:
                    new_holding.current_price = current_price_data['results'][0]['c']
                    new_holding.last_updated = datetime.now()
            elif isinstance(data_service, AlpacaService):
                current_price_data = data_service.get_current_price(holding.ticker.upper())
                if 'price' in current_price_data:
                    new_holding.current_price = current_price_data['price']
                    new_holding.last_updated = datetime.now()
        except Exception as e:
            logger.warning(f"Could not get current price for {holding.ticker}: {str(e)}")
        
        db.add(new_holding)
        db.commit()
        db.refresh(new_holding)
        
        # Calculate additional fields for response
        current_price = new_holding.current_price or new_holding.purchase_price
        current_value = new_holding.quantity * current_price
        profit_loss = current_value - (new_holding.quantity * new_holding.purchase_price)
        profit_loss_pct = profit_loss / (new_holding.quantity * new_holding.purchase_price) if new_holding.quantity * new_holding.purchase_price > 0 else 0
        
        response = new_holding.__dict__.copy()
        response["current_value"] = current_value
        response["profit_loss"] = profit_loss
        response["profit_loss_pct"] = profit_loss_pct
        
        logger.info(f"Added {holding.quantity} shares of {holding.ticker} to portfolio {portfolio_id}")
        return response
    except Exception as e:
        logger.error(f"Error adding holding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolios/{portfolio_id}/holdings", response_model=List[HoldingResponse], tags=["Holdings"])
@log_api_call
async def get_portfolio_holdings(portfolio_id: int, db: Session = Depends(get_db)):
    """Get all holdings for a portfolio"""
    try:
        # Get portfolio
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail=f"Portfolio with ID {portfolio_id} not found")
        
        # Get holdings
        holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
        
        # Calculate additional fields for each holding
        result = []
        for holding in holdings:
            current_price = holding.current_price or holding.purchase_price
            current_value = holding.quantity * current_price
            profit_loss = current_value - (holding.quantity * holding.purchase_price)
            profit_loss_pct = profit_loss / (holding.quantity * holding.purchase_price) if holding.quantity * holding.purchase_price > 0 else 0
            
            holding_dict = holding.__dict__.copy()
            holding_dict["current_value"] = current_value
            holding_dict["profit_loss"] = profit_loss
            holding_dict["profit_loss_pct"] = profit_loss_pct
            result.append(holding_dict)
        
        return result
    except Exception as e:
        logger.error(f"Error getting holdings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
