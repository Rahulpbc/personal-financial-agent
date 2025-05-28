"""
Logging utilities for the financial agent
"""
import logging
import os
import json
from datetime import datetime
from functools import wraps
import time
import traceback
from typing import Callable, Any, Dict

# Create a custom logger
logger = logging.getLogger("financial_agent")

def mask_pii(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mask personally identifiable information in logs (NFR-9)
    
    Args:
        data: Dictionary containing data to mask
        
    Returns:
        Dictionary with masked PII
    """
    # Create a copy to avoid modifying the original
    masked_data = data.copy()
    
    # List of keys that might contain PII
    pii_keys = [
        "username", "name", "email", "address", "phone", "ssn", 
        "social_security", "birth", "dob", "password", "api_key", 
        "secret", "token", "account_number"
    ]
    
    # Mask PII fields
    for key in masked_data:
        if any(pii_term in key.lower() for pii_term in pii_keys):
            if isinstance(masked_data[key], str):
                # Mask all but first and last character
                if len(masked_data[key]) > 2:
                    masked_data[key] = masked_data[key][0] + "*" * (len(masked_data[key]) - 2) + masked_data[key][-1]
                else:
                    masked_data[key] = "**"
    
    return masked_data

def log_api_call(func: Callable) -> Callable:
    """
    Decorator to log API calls (NFR-6)
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Log the API call
        logger.info(f"API Call: {func.__name__}")
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log successful completion
            execution_time = time.time() - start_time
            logger.info(f"API Call Completed: {func.__name__} in {execution_time:.2f}s")
            
            return result
        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            logger.error(f"API Call Failed: {func.__name__} in {execution_time:.2f}s - {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Re-raise the exception
            raise
    
    return wrapper

def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls with timing
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Log the function call
        logger.debug(f"Function Call: {func.__name__}")
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log successful completion
            execution_time = time.time() - start_time
            logger.debug(f"Function Completed: {func.__name__} in {execution_time:.2f}s")
            
            return result
        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            logger.error(f"Function Failed: {func.__name__} in {execution_time:.2f}s - {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Re-raise the exception
            raise
    
    return wrapper

def setup_logger(log_file: str = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with file and console handlers
    
    Args:
        log_file: Path to log file
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logger
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log an error with context
    
    Args:
        error: Exception to log
        context: Additional context for the error
    """
    if context:
        # Mask PII in context
        masked_context = mask_pii(context)
        logger.error(f"Error: {str(error)} - Context: {json.dumps(masked_context)}")
    else:
        logger.error(f"Error: {str(error)}")
    
    # Log traceback at debug level
    logger.debug(f"Traceback: {traceback.format_exc()}")

def handle_api_error(error: Exception, service_name: str) -> Dict[str, Any]:
    """
    Handle API errors gracefully (NFR-7)
    
    Args:
        error: Exception from API call
        service_name: Name of the service
        
    Returns:
        Error response dictionary
    """
    # Log the error
    logger.error(f"{service_name} API Error: {str(error)}")
    
    # Check for specific error types
    if "rate limit" in str(error).lower():
        return {
            "error": "Rate limit exceeded",
            "message": f"The {service_name} API rate limit has been exceeded. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    elif "unauthorized" in str(error).lower() or "authentication" in str(error).lower():
        return {
            "error": "Authentication error",
            "message": f"Failed to authenticate with the {service_name} API. Please check your API credentials.",
            "timestamp": datetime.now().isoformat()
        }
    elif "not found" in str(error).lower():
        return {
            "error": "Resource not found",
            "message": f"The requested resource was not found in the {service_name} API.",
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "error": "Service error",
            "message": f"An error occurred while communicating with the {service_name} API: {str(error)}",
            "timestamp": datetime.now().isoformat()
        }
