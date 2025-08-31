# AI Market Trend Analysis Platform

## Overview

This is a comprehensive AI-powered market trend analysis platform built with Streamlit. The application provides end-to-end capabilities for data collection, time series forecasting, customer segmentation, sentiment analysis, and interactive visualization. It's designed to help businesses analyze market trends, predict future patterns, and derive actionable insights from various data sources including financial markets, customer behavior, and social sentiment.

The platform integrates multiple machine learning approaches including Prophet for time series forecasting, clustering algorithms for customer segmentation, transformer-based models for sentiment analysis, and advanced visualization capabilities for presenting insights.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with custom CSS styling
- **Layout**: Multi-column responsive design with sidebar navigation
- **State Management**: Streamlit session state for maintaining user interactions and loaded data
- **Visualization**: Plotly for interactive charts and graphs with real-time updates

### Backend Architecture
- **Modular Design**: Clean separation of concerns with dedicated modules for each functional area
- **Data Processing Pipeline**: Sequential processing from raw data through preprocessing, analysis, and visualization
- **Model Management**: Centralized model initialization and caching for performance optimization
- **Error Handling**: Comprehensive exception handling with fallback mechanisms

### Machine Learning Components
- **Time Series Forecasting**: Facebook Prophet as primary model with ARIMA, Random Forest, and Linear Regression as alternatives
- **Customer Segmentation**: K-means clustering with RFM analysis, DBSCAN, and Agglomerative clustering options
- **Sentiment Analysis**: Transformer-based models (RoBERTa, DistilBERT) with TextBlob fallback
- **Data Preprocessing**: Automated feature engineering, scaling, encoding, and missing value imputation

### Data Storage Architecture
- **File-based Storage**: CSV files for sample data with automatic generation if files don't exist
- **In-memory Processing**: Pandas DataFrames for data manipulation and analysis
- **Caching Strategy**: Streamlit resource caching for model loading and expensive computations

### Visualization System
- **Interactive Charts**: Plotly-based charts with hover information, zooming, and filtering
- **Dashboard Layout**: Multi-panel dashboard with trend charts, cluster visualizations, and sentiment metrics
- **Customizable Views**: Dynamic chart generation based on user selections and data characteristics

## External Dependencies

### Core ML Libraries
- **transformers**: Hugging Face transformers for pre-trained NLP models
- **prophet**: Facebook Prophet for time series forecasting
- **scikit-learn**: Machine learning algorithms for clustering and preprocessing
- **statsmodels**: Statistical models including ARIMA for time series analysis

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **yfinance**: Yahoo Finance API for stock market data retrieval

### Visualization
- **plotly**: Interactive plotting and dashboard creation
- **streamlit**: Web application framework and UI components
- **seaborn/matplotlib**: Additional plotting capabilities for statistical visualizations

### Text Processing
- **textblob**: Natural language processing and sentiment analysis fallback
- **re**: Regular expressions for text preprocessing

### Additional Integrations
- **requests**: HTTP library for API calls and data fetching
- **warnings**: Python warnings management for cleaner output
- **os**: Environment variable management for API keys and configuration

The system is designed to be extensible with additional data sources, models, and visualization types through its modular architecture.