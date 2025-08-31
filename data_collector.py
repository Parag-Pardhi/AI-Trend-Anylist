import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    """
    Data collection module for market trend analysis.
    Handles various data sources including sample data, APIs, and web scraping.
    """
    
    def __init__(self):
        self.api_key = os.getenv("API_KEY", "demo_key")
        
    def load_sample_data(self):
        """
        Load sample retail data for demonstration and testing.
        Creates realistic market data with seasonal patterns.
        """
        try:
            # Check if sample data file exists
            if os.path.exists('data/sample_retail_data.csv'):
                return pd.read_csv('data/sample_retail_data.csv', parse_dates=['Date'])
            
            # Generate sample data if file doesn't exist
            return self._generate_sample_data()
            
        except Exception as e:
            print(f"Error loading sample data: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate realistic sample market data with trends and seasonality"""
        
        # Date range for 2 years of data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 1, 1)
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        n_days = len(date_range)
        
        # Base sales trend with growth
        base_sales = np.linspace(1000, 1500, n_days)
        
        # Add seasonality (higher sales in Q4, lower in Q1)
        seasonal_factor = np.sin(2 * np.pi * np.arange(n_days) / 365.25) * 200 + 100
        
        # Add weekly pattern (higher sales on weekends)
        weekly_factor = np.sin(2 * np.pi * np.arange(n_days) / 7) * 50
        
        # Add random noise
        noise = np.random.normal(0, 50, n_days)
        
        # Combine all factors
        sales = base_sales + seasonal_factor + weekly_factor + noise
        sales = np.maximum(sales, 0)  # Ensure positive values
        
        # Generate related metrics
        customers = np.random.poisson(sales / 10)  # Customers related to sales
        avg_order_value = sales / customers
        avg_order_value = np.where(customers == 0, 0, avg_order_value)
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        category_weights = np.random.dirichlet([1, 1, 1, 1, 1], n_days)
        
        # Customer demographics
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        age_weights = np.random.dirichlet([0.2, 0.3, 0.25, 0.15, 0.1], n_days)
        
        # Create DataFrame
        sample_data = pd.DataFrame({
            'Date': date_range,
            'Sales': sales.round(2),
            'Customers': customers,
            'Average_Order_Value': avg_order_value.round(2),
            'Electronics_Sales': (sales * category_weights[:, 0]).round(2),
            'Clothing_Sales': (sales * category_weights[:, 1]).round(2),
            'Home_Garden_Sales': (sales * category_weights[:, 2]).round(2),
            'Sports_Sales': (sales * category_weights[:, 3]).round(2),
            'Books_Sales': (sales * category_weights[:, 4]).round(2),
            'Age_18_25_Customers': (customers * age_weights[:, 0]).round(0).astype(int),
            'Age_26_35_Customers': (customers * age_weights[:, 1]).round(0).astype(int),
            'Age_36_45_Customers': (customers * age_weights[:, 2]).round(0).astype(int),
            'Age_46_55_Customers': (customers * age_weights[:, 3]).round(0).astype(int),
            'Age_55_Plus_Customers': (customers * age_weights[:, 4]).round(0).astype(int),
        })
        
        # Add some customer behavior metrics
        sample_data['Return_Rate'] = np.random.uniform(0.05, 0.15, n_days).round(3)
        sample_data['Customer_Satisfaction'] = np.random.uniform(3.5, 5.0, n_days).round(2)
        sample_data['Marketing_Spend'] = (sales * np.random.uniform(0.05, 0.12, n_days)).round(2)
        
        # Set date as index
        sample_data.set_index('Date', inplace=True)
        
        return sample_data
    
    def get_stock_data(self, symbol, period="1y"):
        """
        Fetch stock market data using Yahoo Finance API.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Time period ('1y', '2y', '5y', etc.)
        
        Returns:
            pd.DataFrame: Stock price data with OHLCV
        """
        try:
            ticker = yf.Ticker(symbol)
            stock_data = ticker.history(period=period)
            
            if stock_data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Add additional technical indicators
            stock_data['Returns'] = stock_data['Close'].pct_change()
            stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std()
            stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = stock_data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            stock_data['RSI'] = 100 - (100 / (1 + rs))
            
            return stock_data
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            # Return sample stock data as fallback
            return self._generate_sample_stock_data(symbol)
    
    def _generate_sample_stock_data(self, symbol):
        """Generate sample stock data for demonstration"""
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        
        # Generate realistic stock price movement
        initial_price = 100
        prices = [initial_price]
        
        for i in range(1, len(dates)):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.02)  # 0.1% average daily return, 2% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        # Generate OHLC data
        close_prices = np.array(prices)
        high_prices = close_prices * (1 + np.random.uniform(0, 0.03, len(close_prices)))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.03, len(close_prices)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Volume
        volumes = np.random.lognormal(15, 0.5, len(dates)).astype(int)
        
        stock_data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)
        
        # Add technical indicators
        stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std()
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
        
        return stock_data
    
    def get_economic_indicators(self):
        """
        Fetch economic indicators data.
        In a real implementation, this would connect to economic data APIs.
        """
        try:
            # For demonstration, generate sample economic data
            dates = pd.date_range(end=datetime.now(), periods=60, freq='M')
            
            economic_data = pd.DataFrame({
                'Date': dates,
                'GDP_Growth': np.random.normal(2.5, 1.0, len(dates)),
                'Inflation': np.random.normal(3.0, 0.8, len(dates)),
                'Unemployment': np.random.normal(5.5, 1.5, len(dates)),
                'Interest_Rate': np.random.normal(2.0, 0.5, len(dates)),
                'Consumer_Confidence': np.random.normal(100, 15, len(dates))
            })
            
            economic_data.set_index('Date', inplace=True)
            return economic_data
            
        except Exception as e:
            print(f"Error fetching economic data: {e}")
            return pd.DataFrame()
    
    def collect_social_media_data(self, query, limit=100):
        """
        Collect social media data for sentiment analysis.
        This is a placeholder for social media API integration.
        
        Args:
            query (str): Search query
            limit (int): Number of posts to collect
        
        Returns:
            list: List of text posts
        """
        # Placeholder implementation
        # In real implementation, integrate with Twitter API, Reddit API, etc.
        
        sample_posts = [
            f"Great product! Love the {query} experience!",
            f"Not satisfied with {query}. Poor quality.",
            f"Amazing {query}! Highly recommend to everyone.",
            f"Average {query}, nothing special but works fine.",
            f"Terrible {query} experience. Would not buy again.",
            f"Outstanding {query} quality! Best purchase ever!",
            f"Disappointing {query}. Expected much better.",
            f"Excellent {query} service! Five stars!",
            f"Mediocre {query}. Price is too high for quality.",
            f"Perfect {query}! Exactly what I needed."
        ]
        
        # Return random sample of posts
        import random
        return random.sample(sample_posts, min(limit, len(sample_posts)))
    
    def validate_data_quality(self, df):
        """
        Validate data quality and return quality metrics.
        
        Args:
            df (pd.DataFrame): Data to validate
        
        Returns:
            dict: Data quality metrics
        """
        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'date_columns': len(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
        
        quality_metrics['outliers_by_column'] = outlier_counts
        quality_metrics['total_outliers'] = sum(outlier_counts.values())
        
        return quality_metrics
    
    def preprocess_raw_data(self, df):
        """
        Basic preprocessing for raw data.
        
        Args:
            df (pd.DataFrame): Raw data
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        processed_df = df.copy()
        
        # Handle missing values
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            processed_df[col].fillna(processed_df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            mode_value = processed_df[col].mode().iloc[0] if not processed_df[col].mode().empty else 'Unknown'
            processed_df[col].fillna(mode_value, inplace=True)
        
        # Remove duplicates
        processed_df.drop_duplicates(inplace=True)
        
        # Basic outlier treatment for numeric columns
        for col in numeric_cols:
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return processed_df
