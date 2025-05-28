"""
Financial metrics calculation utilities
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

def calculate_owner_earnings(
    net_income: float,
    depreciation_amortization: float,
    capital_expenditures: float,
    working_capital_change: float
) -> float:
    """
    Calculate Owner Earnings
    
    Owner Earnings = Net Income + Depreciation & Amortization - Capital Expenditures - Change in Working Capital
    
    Args:
        net_income: Net income from income statement
        depreciation_amortization: Depreciation and amortization from cash flow statement
        capital_expenditures: Capital expenditures from cash flow statement
        working_capital_change: Change in working capital from cash flow statement
        
    Returns:
        Owner earnings value
    """
    return net_income + depreciation_amortization - capital_expenditures - working_capital_change

def calculate_roe(net_income: float, shareholders_equity: float) -> float:
    """
    Calculate Return on Equity (ROE)
    
    ROE = Net Income / Shareholders' Equity
    
    Args:
        net_income: Net income from income statement
        shareholders_equity: Total shareholders' equity from balance sheet
        
    Returns:
        ROE as a decimal value
    """
    if shareholders_equity == 0:
        return 0
    return net_income / shareholders_equity

def calculate_roic(
    net_operating_profit_after_tax: float,
    total_debt: float,
    shareholders_equity: float,
    cash_and_equivalents: float
) -> float:
    """
    Calculate Return on Invested Capital (ROIC)
    
    ROIC = NOPAT / Invested Capital
    where Invested Capital = Total Debt + Shareholders' Equity - Cash and Equivalents
    
    Args:
        net_operating_profit_after_tax: NOPAT from income statement
        total_debt: Total debt from balance sheet
        shareholders_equity: Total shareholders' equity from balance sheet
        cash_and_equivalents: Cash and cash equivalents from balance sheet
        
    Returns:
        ROIC as a decimal value
    """
    invested_capital = total_debt + shareholders_equity - cash_and_equivalents
    
    if invested_capital == 0:
        return 0
    
    return net_operating_profit_after_tax / invested_capital

def calculate_dcf_valuation(
    cash_flows: List[float],
    terminal_growth_rate: float,
    discount_rate: float,
    shares_outstanding: float
) -> Dict[str, Any]:
    """
    Perform a simple Discounted Cash Flow (DCF) valuation
    
    Args:
        cash_flows: List of projected future cash flows
        terminal_growth_rate: Expected long-term growth rate after projection period
        discount_rate: Required rate of return (WACC)
        shares_outstanding: Number of shares outstanding
        
    Returns:
        Dictionary containing:
        - present_value: Present value of projected cash flows
        - terminal_value: Terminal value
        - enterprise_value: Enterprise value (PV + TV)
        - equity_value: Equity value
        - fair_value_per_share: Fair value per share
    """
    # Calculate present value of projected cash flows
    present_value = 0
    for i, cf in enumerate(cash_flows):
        present_value += cf / ((1 + discount_rate) ** (i + 1))
    
    # Calculate terminal value using perpetuity growth model
    last_cash_flow = cash_flows[-1]
    terminal_value = last_cash_flow * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    
    # Discount terminal value to present
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** len(cash_flows))
    
    # Calculate enterprise value
    enterprise_value = present_value + discounted_terminal_value
    
    # Equity value (simplified, assuming no debt or cash adjustments)
    equity_value = enterprise_value
    
    # Calculate fair value per share
    fair_value_per_share = equity_value / shares_outstanding
    
    return {
        "present_value": present_value,
        "terminal_value": terminal_value,
        "discounted_terminal_value": discounted_terminal_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value_per_share": fair_value_per_share
    }

def extract_financial_data_from_polygon(financial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant financial data from Polygon API response
    
    Args:
        financial_data: Financial data from Polygon API
        
    Returns:
        Dictionary with extracted financial metrics
    """
    results = {}
    
    try:
        financials = financial_data.get("results", [])
        if not financials:
            return {"error": "No financial data available"}
        
        latest_report = financials[0]
        
        # Extract income statement data
        income_statement = latest_report.get("financials", {}).get("income_statement", {})
        results["net_income"] = income_statement.get("net_income_loss", 0)
        results["revenue"] = income_statement.get("revenues", 0)
        results["operating_income"] = income_statement.get("operating_income_loss", 0)
        
        # Extract balance sheet data
        balance_sheet = latest_report.get("financials", {}).get("balance_sheet", {})
        results["total_assets"] = balance_sheet.get("assets", 0)
        results["total_liabilities"] = balance_sheet.get("liabilities", 0)
        results["shareholders_equity"] = balance_sheet.get("stockholders_equity", 0)
        results["cash_and_equivalents"] = balance_sheet.get("cash_and_equivalents", 0)
        results["total_debt"] = balance_sheet.get("debt", 0)
        
        # Extract cash flow statement data
        cash_flow = latest_report.get("financials", {}).get("cash_flow_statement", {})
        results["operating_cash_flow"] = cash_flow.get("net_cash_flow_from_operating_activities", 0)
        results["capital_expenditures"] = cash_flow.get("payments_to_acquire_property_plant_equipment", 0)
        results["depreciation_amortization"] = cash_flow.get("depreciation_depletion_and_amortization", 0)
        
        # Calculate working capital change (simplified)
        results["working_capital_change"] = 0  # Would need more data for accurate calculation
        
        # Calculate additional metrics
        results["owner_earnings"] = calculate_owner_earnings(
            results["net_income"],
            results["depreciation_amortization"],
            results["capital_expenditures"],
            results["working_capital_change"]
        )
        
        results["roe"] = calculate_roe(
            results["net_income"],
            results["shareholders_equity"]
        )
        
        # Simplified NOPAT calculation
        tax_rate = 0.21  # Assumed corporate tax rate
        nopat = results["operating_income"] * (1 - tax_rate)
        
        results["roic"] = calculate_roic(
            nopat,
            results["total_debt"],
            results["shareholders_equity"],
            results["cash_and_equivalents"]
        )
        
        # Add period information
        results["fiscal_year"] = latest_report.get("fiscal_year", "")
        results["fiscal_period"] = latest_report.get("fiscal_period", "")
        
        return results
    
    except Exception as e:
        return {"error": f"Error extracting financial data: {str(e)}"}
