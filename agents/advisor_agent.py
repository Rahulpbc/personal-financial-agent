"""
RAG-driven financial advisor agent using LangChain
"""
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.chains import LLMChain

from services.news_service import NewsService
from services.alpaca_service import AlpacaService
from services.polygon_service import PolygonService
from utils.financial_metrics import extract_financial_data_from_polygon
from data.database.models import User, Portfolio

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AdvisorAgent:
    """RAG-driven financial advisor agent using LangChain"""
    
    def __init__(self, use_polygon: bool = True):
        """
        Initialize the advisor agent
        
        Args:
            use_polygon: Whether to use Polygon API (True) or Alpaca API (False)
        """
        # Initialize API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            logger.error("OpenAI API key not found")
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
        
        # Initialize services
        self.news_service = NewsService()
        
        if use_polygon:
            self.data_service = PolygonService()
            logger.info("Using Polygon API for financial data")
        else:
            self.data_service = AlpacaService()
            logger.info("Using Alpaca API for financial data")
        
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Create agent
        self.agent_executor = self._create_agent()
        
        logger.info("Advisor agent initialized")
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools"""
        tools = [
            self.get_current_price,
            self.get_relevant_news,
            self.get_financial_metrics,
            self.get_market_sentiment
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial advisor assistant that helps users make informed investment decisions.
            You have access to real-time financial data, news, and market sentiment information.
            
            When providing advice:
            1. Base your recommendations on the retrieved financial data and news
            2. Consider the user's profile (age, investment horizon, risk tolerance)
            3. Provide clear explanations of your reasoning
            4. Always cite your sources (news articles, financial data)
            5. Be transparent about uncertainties and risks
            
            Important guidelines:
            - Do not make specific buy/sell recommendations
            - Frame advice in terms of considerations and factors to evaluate
            - Acknowledge the limitations of your analysis
            - Emphasize the importance of diversification and long-term investing
            - Remind users that past performance does not guarantee future results
            
            Always maintain a professional tone and provide educational context.
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
            result = self.data_service.get_current_price(ticker)
            logger.info(f"Retrieved current price for {ticker}")
            return {
                "ticker": ticker,
                "price_data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    @tool
    def get_relevant_news(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Get relevant news articles for a query.
        
        Args:
            query: Search query (can be a ticker symbol or topic)
            k: Number of articles to retrieve
            
        Returns:
            Dictionary with relevant news articles
        """
        try:
            # Search for relevant articles
            articles = self.news_service.search(query, k=k)
            
            # If no articles found, fetch new ones
            if not articles:
                logger.info(f"No articles found for {query}, fetching new ones")
                
                # Check if query looks like a ticker symbol
                if len(query) <= 5 and query.isupper():
                    self.news_service.fetch_and_index_for_tickers([query])
                else:
                    self.news_service.fetch_and_index_articles(
                        self.news_service.fetch_news(query=query)
                    )
                
                # Try searching again
                articles = self.news_service.search(query, k=k)
            
            logger.info(f"Retrieved {len(articles)} relevant news articles for {query}")
            
            # Format articles for the agent
            formatted_articles = []
            for article in articles:
                formatted_articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", ""),
                    "published_at": article.get("published_at", ""),
                    "url": article.get("url", ""),
                    "relevance_score": article.get("relevance_score", 0)
                })
            
            return {
                "query": query,
                "articles": formatted_articles,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting relevant news for {query}: {str(e)}")
            return {"error": str(e)}
    
    @tool
    def get_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        Get financial metrics for a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., AAPL)
            
        Returns:
            Dictionary with financial metrics
        """
        try:
            # Get financial data
            if isinstance(self.data_service, PolygonService):
                financial_data = self.data_service.get_financial_data(ticker)
                metrics = extract_financial_data_from_polygon(financial_data)
            else:
                # For Alpaca, we don't have detailed financials, so return basic info
                current_price = self.data_service.get_current_price(ticker)
                metrics = {
                    "current_price": current_price.get("price", 0),
                    "note": "Detailed financial metrics not available through Alpaca API"
                }
            
            logger.info(f"Retrieved financial metrics for {ticker}")
            
            return {
                "ticker": ticker,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting financial metrics for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    @tool
    def get_market_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Get market sentiment for a ticker symbol based on recent news.
        
        Args:
            ticker: Stock ticker symbol (e.g., AAPL)
            
        Returns:
            Dictionary with market sentiment analysis
        """
        try:
            # Get relevant news
            articles = self.news_service.search(ticker, k=10)
            
            # If no articles found, fetch new ones
            if not articles:
                logger.info(f"No articles found for {ticker}, fetching new ones")
                self.news_service.fetch_and_index_for_tickers([ticker])
                articles = self.news_service.search(ticker, k=10)
            
            # Extract titles and descriptions for sentiment analysis
            texts = []
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                texts.append(f"{title} {description}")
            
            # Use LLM for sentiment analysis
            if texts:
                sentiment_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a financial sentiment analyzer. 
                    Analyze the following news articles about a company or stock and determine the overall market sentiment.
                    Rate the sentiment on a scale from -5 (extremely negative) to +5 (extremely positive).
                    Provide a brief explanation of your rating and identify key themes or trends in the news.
                    """),
                    ("human", f"News articles about {ticker}:\n\n" + "\n\n".join(texts))
                ])
                
                sentiment_chain = LLMChain(llm=self.llm, prompt=sentiment_prompt)
                sentiment_result = sentiment_chain.run("")
                
                logger.info(f"Analyzed market sentiment for {ticker}")
                
                return {
                    "ticker": ticker,
                    "sentiment_analysis": sentiment_result,
                    "based_on": len(articles),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "ticker": ticker,
                    "sentiment_analysis": "No recent news articles found to analyze sentiment.",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error analyzing market sentiment for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    async def generate_advice(self, user: User, query: str) -> Dict[str, Any]:
        """
        Generate personalized financial advice based on user profile and query
        
        Args:
            user: User object with profile information
            query: User's question or request
            
        Returns:
            Dictionary with advice and supporting information
        """
        try:
            # Format user profile for context
            user_profile = f"""
            User Profile:
            - Age: {user.age}
            - Investment Horizon: {user.investment_horizon} years
            - Risk Tolerance: {user.risk_tolerance}
            - Target Return: {user.target_return:.2%}
            """
            
            # Combine user profile with query
            input_text = f"{user_profile}\n\nUser Query: {query}"
            
            # Get advice from agent
            result = await self.agent_executor.ainvoke({"input": input_text})
            
            logger.info(f"Generated advice for user {user.username}")
            
            return {
                "username": user.username,
                "query": query,
                "advice": result["output"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating advice: {str(e)}")
            raise
