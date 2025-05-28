"""
SQLAlchemy database models for the financial agent
"""
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create base class for SQLAlchemy models
Base = declarative_base()

class User(Base):
    """User profile model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    age = Column(Integer, nullable=False)
    investment_horizon = Column(Integer, nullable=False)  # in years
    risk_tolerance = Column(String(20), nullable=False)  # low, medium, high
    target_return = Column(Float, nullable=False)  # annual percentage
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', age={self.age}, risk_tolerance='{self.risk_tolerance}')>"

class Portfolio(Base):
    """Portfolio model"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Portfolio(name='{self.name}', user_id={self.user_id})>"

class Holding(Base):
    """Portfolio holding model"""
    __tablename__ = "holdings"
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    purchase_date = Column(DateTime, nullable=False)
    current_price = Column(Float, nullable=True)
    last_updated = Column(DateTime, nullable=True)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")
    
    def __repr__(self):
        return f"<Holding(ticker='{self.ticker}', quantity={self.quantity}, portfolio_id={self.portfolio_id})>"

class StockPrice(Base):
    """Stock price model for storing historical and current prices"""
    __tablename__ = "stock_prices"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    def __repr__(self):
        return f"<StockPrice(ticker='{self.ticker}', date='{self.date}', close_price={self.close_price})>"

class NewsArticle(Base):
    """News article model for storing news data"""
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    url = Column(String(255), nullable=False)
    source = Column(String(100), nullable=False)
    published_at = Column(DateTime, nullable=False)
    tickers = Column(String(255), nullable=True)  # Comma-separated list of related tickers
    embedding_id = Column(String(100), nullable=True)  # ID for retrieving embedding from FAISS
    
    def __repr__(self):
        return f"<NewsArticle(title='{self.title}', published_at='{self.published_at}')>"

class BacktestResult(Base):
    """Backtest result model for storing backtest data"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_value = Column(Float, nullable=False)
    final_value = Column(Float, nullable=False)
    return_pct = Column(Float, nullable=False)
    benchmark_return_pct = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<BacktestResult(portfolio_id={self.portfolio_id}, return_pct={self.return_pct})>"

class TradeLog(Base):
    """Trade log model for storing simulated trades"""
    __tablename__ = "trade_logs"
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String(20), nullable=False)
    trade_date = Column(DateTime, nullable=False)
    action = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<TradeLog(ticker='{self.ticker}', action='{self.action}', quantity={self.quantity})>"

def get_engine(db_path=None):
    """Get SQLAlchemy engine"""
    if db_path is None:
        db_path = os.getenv("DATABASE_URL", "sqlite:///./financial_agent.db")
    
    return create_engine(db_path)

def init_db(engine=None):
    """Initialize database tables"""
    if engine is None:
        engine = get_engine()
    
    Base.metadata.create_all(engine)
