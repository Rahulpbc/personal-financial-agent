# Personal Financial Agent

A financial analysis agent built with Langchain and FastAPI that leverages the Polygon API to access financial data and perform various financial calculations.

## Features

- Access current stock prices via Polygon API
- Retrieve historical price data for financial analysis
- Fetch latest news related to specific tickers
- Access fundamental financial data for companies
- Compute key financial metrics:
  - Owner Earnings
  - Return on Equity (ROE)
  - Return on Invested Capital (ROIC)
- Perform discounted cash flow (DCF) valuation

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your Polygon API key:
   ```
   POLYGON_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```
4. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

## API Endpoints

The API documentation will be available at `http://localhost:8000/docs` after starting the application.

## Project Structure

```
personal-financial-agent/
├── app/                # FastAPI application
├── data/               # Data models and storage
├── models/             # LangChain models
├── services/           # Services for external APIs
├── utils/              # Utility functions
└── tests/              # Test cases
```