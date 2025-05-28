"""
Data models for the financial agent
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime
from enum import Enum


class TickerRequest(BaseModel):
    """Request model for ticker-related operations"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")


class HistoricalDataRequest(BaseModel):
    """Request model for historical data"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    from_date: str = Field(..., description="Start date in format YYYY-MM-DD")
    to_date: str = Field(..., description="End date in format YYYY-MM-DD")
    timespan: str = Field("day", description="Time span (day, hour, minute, etc.)")


class NewsRequest(BaseModel):
    """Request model for news data"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    limit: int = Field(10, description="Number of news items to return")


class FinancialDataRequest(BaseModel):
    """Request model for financial data"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    timeframe: str = Field("annual", description="Timeframe for financial data (annual, quarterly)")


class DCFValuationRequest(BaseModel):
    """Request model for DCF valuation"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    projected_growth_rates: List[float] = Field(..., description="Projected growth rates for future cash flows")
    terminal_growth_rate: float = Field(0.02, description="Expected long-term growth rate after projection period")
    discount_rate: float = Field(0.1, description="Required rate of return (WACC)")
    projection_years: int = Field(5, description="Number of years to project cash flows")


class FinancialMetricsResponse(BaseModel):
    """Response model for financial metrics"""
    ticker: str
    metrics: Dict[str, Any]
    timestamp: str


class DCFValuationResponse(BaseModel):
    """Response model for DCF valuation"""
    ticker: str
    valuation: Dict[str, Any]
    assumptions: Dict[str, Any]
    timestamp: str


class AgentQueryRequest(BaseModel):
    """Request model for agent queries"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    query: str = Field(..., description="Natural language query about the ticker")


# New models for enhanced requirements
class RiskToleranceEnum(str, Enum):
    """Risk tolerance levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UserProfileCreate(BaseModel):
    """Request model for creating a user profile"""
    username: str = Field(..., description="Username")
    age: int = Field(..., description="User's age")
    investment_horizon: int = Field(..., description="Investment horizon in years")
    risk_tolerance: RiskToleranceEnum = Field(..., description="Risk tolerance (low, medium, high)")
    target_return: float = Field(..., description="Target annual return as decimal (e.g., 0.08 for 8%)")
    
    @validator('age')
    def validate_age(cls, v):
        if v < 18 or v > 120:
            raise ValueError('Age must be between 18 and 120')
        return v
    
    @validator('investment_horizon')
    def validate_investment_horizon(cls, v):
        if v < 1 or v > 50:
            raise ValueError('Investment horizon must be between 1 and 50 years')
        return v
    
    @validator('target_return')
    def validate_target_return(cls, v):
        if v < 0 or v > 0.5:
            raise ValueError('Target return must be between 0 and 0.5 (0% to 50%)')
        return v


class UserProfileResponse(BaseModel):
    """Response model for user profile"""
    id: int
    username: str
    age: int
    investment_horizon: int
    risk_tolerance: str
    target_return: float
    created_at: datetime
    updated_at: datetime


class PortfolioCreate(BaseModel):
    """Request model for creating a portfolio"""
    name: str = Field(..., description="Portfolio name")
    description: Optional[str] = Field(None, description="Portfolio description")


class PortfolioResponse(BaseModel):
    """Response model for portfolio"""
    id: int
    user_id: int
    name: str
    description: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    total_value: Optional[float] = None
    holdings_count: Optional[int] = None


class HoldingCreate(BaseModel):
    """Request model for creating a holding"""
    ticker: str = Field(..., description="Stock ticker symbol")
    quantity: float = Field(..., description="Number of shares")
    purchase_price: float = Field(..., description="Purchase price per share")
    purchase_date: date = Field(..., description="Purchase date")


class HoldingResponse(BaseModel):
    """Response model for holding"""
    id: int
    portfolio_id: int
    ticker: str
    quantity: float
    purchase_price: float
    purchase_date: date
    current_price: Optional[float]
    current_value: Optional[float]
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    last_updated: Optional[datetime]


class PortfolioOptimizationRequest(BaseModel):
    """Request model for portfolio optimization"""
    portfolio_id: int = Field(..., description="Portfolio ID")
    risk_tolerance: Optional[RiskToleranceEnum] = Field(None, description="Risk tolerance override")
    target_return: Optional[float] = Field(None, description="Target annual return override")


class PortfolioOptimizationResponse(BaseModel):
    """Response model for portfolio optimization"""
    portfolio_id: int
    current_weights: Dict[str, float]
    optimized_weights: Dict[str, float]
    performance: Dict[str, float]
    allocation: Dict[str, int]
    chart_url: Optional[str]


class BacktestRequest(BaseModel):
    """Request model for portfolio backtesting"""
    portfolio_id: int = Field(..., description="Portfolio ID")
    months: int = Field(6, description="Number of months to backtest")
    benchmark: str = Field("SPY", description="Benchmark ticker symbol")


class BacktestResponse(BaseModel):
    """Response model for backtest results"""
    portfolio_id: int
    start_date: date
    end_date: date
    initial_value: float
    final_value: float
    return_pct: float
    benchmark_return_pct: float
    metrics: Dict[str, float]
    chart_url: Optional[str]
    trades_url: Optional[str]


class AdvisorQueryRequest(BaseModel):
    """Request model for advisor queries"""
    username: str = Field(..., description="Username")
    query: str = Field(..., description="Natural language query for financial advice")


class AdvisorQueryResponse(BaseModel):
    """Response model for advisor queries"""
    username: str
    query: str
    advice: str
    sources: List[Dict[str, Any]]
    timestamp: str
