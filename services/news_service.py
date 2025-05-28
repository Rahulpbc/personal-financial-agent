"""
News API Service for fetching and indexing financial news
"""
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import logging
from newsapi import NewsApiClient

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class NewsService:
    """Service for fetching and indexing financial news"""
    
    def __init__(self, max_articles: int = 10000):
        """
        Initialize the news service
        
        Args:
            max_articles: Maximum number of articles to store (NFR-5)
        """
        self.api_key = os.getenv("NEWS_API_KEY")
        self.max_articles = max_articles
        
        if not self.api_key:
            logger.error("News API key not found")
            raise ValueError("News API key not found. Please set NEWS_API_KEY in .env file.")
        
        self.newsapi = NewsApiClient(api_key=self.api_key)
        self.index_path = os.path.join(os.getcwd(), "data", "news_index.faiss")
        self.vectorizer_path = os.path.join(os.getcwd(), "data", "news_vectorizer.pkl")
        self.articles_path = os.path.join(os.getcwd(), "data", "news_articles.pkl")
        
        # Initialize or load FAISS index
        self._init_faiss_index()
        
        logger.info("News service initialized")
    
    def _init_faiss_index(self):
        """Initialize or load FAISS index"""
        try:
            # Check if index already exists
            if os.path.exists(self.index_path) and os.path.exists(self.vectorizer_path) and os.path.exists(self.articles_path):
                # Load existing index and vectorizer
                self.index = faiss.read_index(self.index_path)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(self.articles_path, 'rb') as f:
                    self.articles = pickle.load(f)
                
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} articles")
            else:
                # Create new index and vectorizer
                self.vectorizer = TfidfVectorizer(stop_words='english')
                self.articles = []
                
                # Initialize empty index
                self.index = None
                
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            # Create new index and vectorizer as fallback
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.articles = []
            self.index = None
            logger.info("Created new FAISS index (fallback)")
    
    def _save_index(self):
        """Save FAISS index, vectorizer, and articles to disk"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, self.index_path)
            
            # Save vectorizer
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save articles
            with open(self.articles_path, 'wb') as f:
                pickle.dump(self.articles, f)
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} articles")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    def fetch_news(self, query: str = None, tickers: List[str] = None, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch news articles
        
        Args:
            query: Search query
            tickers: List of ticker symbols
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Format dates for NewsAPI
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')
            
            # Build query string
            if tickers and not query:
                # Join tickers with OR for NewsAPI
                query = " OR ".join(tickers)
            elif tickers and query:
                # Combine query with tickers
                ticker_query = " OR ".join(tickers)
                query = f"({query}) AND ({ticker_query})"
            
            # Default to financial news if no query or tickers
            if not query:
                query = "finance OR stock OR market OR investing"
            
            # Fetch articles
            response = self.newsapi.get_everything(
                q=query,
                from_param=from_date_str,
                to=to_date_str,
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            logger.info(f"Fetched {len(response['articles'])} news articles for query: {query}")
            
            # Process articles
            articles = []
            for article in response['articles']:
                articles.append({
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'url': article['url'],
                    'source': article['source']['name'],
                    'published_at': article['publishedAt'],
                    'tickers': ",".join(tickers) if tickers else None
                })
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            raise
    
    def index_articles(self, articles: List[Dict[str, Any]]) -> None:
        """
        Index articles in FAISS
        
        Args:
            articles: List of news articles to index
        """
        try:
            if not articles:
                logger.warning("No articles to index")
                return
            
            # Extract text content for vectorization
            texts = []
            for article in articles:
                # Combine title, description, and content for better embeddings
                text = f"{article['title']} {article['description'] or ''} {article['content'] or ''}"
                texts.append(text)
            
            # Check if we need to initialize the vectorizer with the first batch
            if not self.articles:
                # Fit vectorizer on the first batch of texts
                X = self.vectorizer.fit_transform(texts).astype('float32')
                
                # Initialize FAISS index
                d = X.shape[1]  # Dimensionality of the vectors
                self.index = faiss.IndexFlatL2(d)
                
                logger.info(f"Initialized FAISS index with dimension {d}")
            else:
                # Transform new texts using existing vectorizer
                X = self.vectorizer.transform(texts).astype('float32')
            
            # Convert sparse matrix to dense
            X_dense = X.toarray()
            
            # Add vectors to the index
            self.index.add(X_dense)
            
            # Store articles
            self.articles.extend(articles)
            
            # Enforce maximum article limit (NFR-5)
            if len(self.articles) > self.max_articles:
                # Remove oldest articles
                excess = len(self.articles) - self.max_articles
                self.articles = self.articles[excess:]
                
                # Rebuild index with remaining articles
                texts = []
                for article in self.articles:
                    text = f"{article['title']} {article['description'] or ''} {article['content'] or ''}"
                    texts.append(text)
                
                X = self.vectorizer.transform(texts).astype('float32')
                X_dense = X.toarray()
                
                # Reset index
                d = X_dense.shape[1]
                self.index = faiss.IndexFlatL2(d)
                self.index.add(X_dense)
                
                logger.info(f"Pruned index to {self.max_articles} articles")
            
            # Save index to disk
            self._save_index()
            
            logger.info(f"Indexed {len(articles)} articles, total: {len(self.articles)}")
        except Exception as e:
            logger.error(f"Error indexing articles: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for articles related to a query
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant articles
        """
        try:
            if not self.index or self.index.ntotal == 0:
                logger.warning("No articles indexed yet")
                return []
            
            # Transform query using vectorizer
            q_vector = self.vectorizer.transform([query]).astype('float32')
            q_vector_dense = q_vector.toarray()
            
            # Search index
            D, I = self.index.search(q_vector_dense, k)
            
            # Get results
            results = []
            for i in range(len(I[0])):
                if I[0][i] < len(self.articles):
                    article = self.articles[I[0][i]]
                    # Add distance score
                    article_with_score = article.copy()
                    article_with_score['relevance_score'] = float(D[0][i])
                    results.append(article_with_score)
            
            logger.info(f"Found {len(results)} articles for query: {query}")
            
            return results
        except Exception as e:
            logger.error(f"Error searching articles: {str(e)}")
            raise
    
    def fetch_and_index_for_tickers(self, tickers: List[str], days: int = 7) -> None:
        """
        Fetch and index news for a list of tickers
        
        Args:
            tickers: List of ticker symbols
            days: Number of days to look back
        """
        try:
            # Fetch news for tickers
            articles = self.fetch_news(tickers=tickers, days=days)
            
            # Index articles
            self.index_articles(articles)
            
            logger.info(f"Fetched and indexed news for tickers: {tickers}")
        except Exception as e:
            logger.error(f"Error fetching and indexing news for tickers: {str(e)}")
            raise
