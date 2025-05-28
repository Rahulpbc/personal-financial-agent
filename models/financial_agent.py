"""
Financial agent using LangChain for natural language processing of financial queries
"""
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import AIMessage, HumanMessage

from services.polygon_service import PolygonService
from utils.financial_metrics import extract_financial_data_from_polygon, calculate_dcf_valuation

# Load environment variables
load_dotenv()

class FinancialAgent:
    """LangChain-based financial agent for processing natural language queries"""
    
    def __init__(self):
        """Initialize the financial agent with necessary components"""
        self.polygon_service = PolygonService()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.memory = ConversationBufferMemory(return_messages=True)
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools"""
        tools = [
            self.get_current_price,
            self.get_historical_prices,
            self.get_ticker_news,
            self.get_financial_metrics,
            self.perform_dcf_valuation
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analysis assistant that helps users analyze stocks and financial data.
            You have access to real-time and historical financial data through the Polygon API.
            You can compute financial metrics like owner earnings, return on equity (ROE), and return on invested capital (ROIC).
            You can also perform discounted cash flow (DCF) valuations.
            
            When answering questions:
            1. Use the appropriate tools to gather the necessary data
            2. Provide clear explanations of financial concepts
            3. Be precise with numbers and calculations
            4. Cite the source and time of the data
            
            Always maintain a professional tone and avoid making investment recommendations.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )
    
    @tool
    def get_current_price(self, ticker: str) -> Dict[str, Any]:
        """
        Get the current price for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., AAPL)
            
        Returns:
            Dictionary with current price information
        """
        try:
            result = self.polygon_service.get_current_price(ticker)
            return {
                "ticker": ticker,
                "price_data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @tool
    def get_historical_prices(self, ticker: str, from_date: str, to_date: str, timespan: str = "day") -> Dict[str, Any]:
        """
        Get historical prices for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., AAPL)
            from_date: Start date in format YYYY-MM-DD
            to_date: End date in format YYYY-MM-DD
            timespan: Time span (day, hour, minute, etc.)
            
        Returns:
            Dictionary with historical price information
        """
        try:
            result = self.polygon_service.get_historical_prices(ticker, from_date, to_date, timespan)
            return {
                "ticker": ticker,
                "historical_data": result,
                "parameters": {
                    "from_date": from_date,
                    "to_date": to_date,
                    "timespan": timespan
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @tool
    def get_ticker_news(self, ticker: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get latest news for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., AAPL)
            limit: Number of news items to return
            
        Returns:
            Dictionary with news information
        """
        try:
            result = self.polygon_service.get_ticker_news(ticker, limit)
            return {
                "ticker": ticker,
                "news_data": result,
                "parameters": {
                    "limit": limit
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @tool
    def get_financial_metrics(self, ticker: str, timeframe: str = "annual") -> Dict[str, Any]:
        """
        Get financial metrics for a ticker symbol including owner earnings, ROE, and ROIC.
        
        Args:
            ticker: Stock ticker symbol (e.g., AAPL)
            timeframe: Timeframe for financial data (annual, quarterly)
            
        Returns:
            Dictionary with financial metrics
        """
        try:
            # Get financial data from Polygon API
            financial_data = self.polygon_service.get_financial_data(ticker, timeframe)
            
            # Extract and calculate metrics
            metrics = extract_financial_data_from_polygon(financial_data)
            
            return {
                "ticker": ticker,
                "metrics": metrics,
                "parameters": {
                    "timeframe": timeframe
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @tool
    def perform_dcf_valuation(
        self, 
        ticker: str, 
        projected_growth_rates: List[float], 
        terminal_growth_rate: float = 0.02,
        discount_rate: float = 0.1,
        projection_years: int = 5
    ) -> Dict[str, Any]:
        """
        Perform a discounted cash flow (DCF) valuation for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., AAPL)
            projected_growth_rates: List of projected growth rates for future cash flows
            terminal_growth_rate: Expected long-term growth rate after projection period
            discount_rate: Required rate of return (WACC)
            projection_years: Number of years to project cash flows
            
        Returns:
            Dictionary with DCF valuation results
        """
        try:
            # Get financial data from Polygon API
            financial_data = self.polygon_service.get_financial_data(ticker, "annual")
            
            # Extract financial metrics
            metrics = extract_financial_data_from_polygon(financial_data)
            
            # Get ticker details to get shares outstanding
            ticker_details = self.polygon_service.get_ticker_details(ticker)
            shares_outstanding = ticker_details.get("results", {}).get("weighted_shares_outstanding", 0)
            
            # Use free cash flow as the base for projection
            # Simplified: FCF = Operating Cash Flow - Capital Expenditures
            base_cash_flow = metrics.get("operating_cash_flow", 0) - metrics.get("capital_expenditures", 0)
            
            # Project future cash flows
            projected_cash_flows = []
            current_cf = base_cash_flow
            
            for i in range(projection_years):
                growth_rate = projected_growth_rates[i] if i < len(projected_growth_rates) else projected_growth_rates[-1]
                current_cf = current_cf * (1 + growth_rate)
                projected_cash_flows.append(current_cf)
            
            # Perform DCF valuation
            valuation = calculate_dcf_valuation(
                projected_cash_flows,
                terminal_growth_rate,
                discount_rate,
                shares_outstanding
            )
            
            return {
                "ticker": ticker,
                "valuation": valuation,
                "assumptions": {
                    "base_cash_flow": base_cash_flow,
                    "projected_growth_rates": projected_growth_rates,
                    "terminal_growth_rate": terminal_growth_rate,
                    "discount_rate": discount_rate,
                    "projection_years": projection_years,
                    "shares_outstanding": shares_outstanding
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def process_query(self, ticker: str, query: str) -> Dict[str, Any]:
        """
        Process a natural language query about a ticker
        
        Args:
            ticker: Stock ticker symbol
            query: Natural language query about the ticker
            
        Returns:
            Agent's response
        """
        input_text = f"Ticker: {ticker}. Query: {query}"
        result = await self.agent_executor.ainvoke({"input": input_text})
        
        return {
            "ticker": ticker,
            "query": query,
            "response": result["output"],
            "timestamp": datetime.now().isoformat()
        }
