import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_collector import DataCollector
from forecasting_models import TimeSeriesForecaster
from clustering_analysis import CustomerSegmentation
from sentiment_analyzer import MarketSentimentAnalyzer
from utils.visualization import create_trend_chart, create_cluster_visualization
from utils.data_preprocessor import DataPreprocessor

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 AI Market Trend Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Main title
st.markdown('<h1 class="main-header">🚀 AI Market Trend Analysis System</h1>', unsafe_allow_html=True)
st.markdown("**Advanced AI-powered market analysis with forecasting, clustering, and sentiment insights**")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["🏠 Home", "📈 Market Forecasting", "👥 Customer Segmentation", "💭 Sentiment Analysis", "📊 Complete Dashboard"]
)

# Initialize components
@st.cache_resource
def load_components():
    """Load and cache all AI components"""
    data_collector = DataCollector()
    forecaster = TimeSeriesForecaster()
    segmentation = CustomerSegmentation()
    sentiment_analyzer = MarketSentimentAnalyzer()
    preprocessor = DataPreprocessor()
    return data_collector, forecaster, segmentation, sentiment_analyzer, preprocessor

try:
    data_collector, forecaster, segmentation, sentiment_analyzer, preprocessor = load_components()
    st.sidebar.success("✅ AI Components Loaded!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading components: {str(e)}")
    st.stop()

# Home page
if page == "🏠 Home":
    st.header("Welcome to AI Market Trend Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        ### 📈 Market Forecasting
        - Time series analysis using Prophet & ARIMA
        - Demand prediction and trend identification
        - Sales forecasting with confidence intervals
        """)
    
    with col2:
        st.info("""
        ### 👥 Customer Segmentation
        - K-Means clustering analysis
        - Behavioral pattern identification
        - RFM analysis for customer value
        """)
    
    with col3:
        st.info("""
        ### 💭 Sentiment Analysis
        - Market sentiment from social data
        - Product review analysis
        - Brand perception insights
        """)
    
    # Sample data upload section
    st.subheader("📁 Data Upload & Management")
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.markdown("#### Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload your market data (sales, customer, product information)"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                st.session_state.uploaded_data = df
                st.session_state.data_loaded = True
                
                with st.expander("Preview Data"):
                    st.dataframe(df.head())
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    with col_upload2:
        st.markdown("#### Or Use Sample Dataset")
        if st.button("📊 Load Sample Retail Data", type="primary"):
            try:
                sample_data = data_collector.load_sample_data()
                st.session_state.sample_data = sample_data
                st.session_state.data_loaded = True
                st.success("✅ Sample data loaded successfully!")
                
                with st.expander("Preview Sample Data"):
                    st.dataframe(sample_data.head())
                    
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    # Enhanced Market data with Indian market support
    st.subheader("🌐 Real-time Market Data")
    
    # Market selection tabs
    market_tab1, market_tab2 = st.tabs(["🇺🇸 Global Markets", "🇮🇳 Indian Markets"])
    
    with market_tab1:
        st.markdown("#### Global Stock Markets")
        col1, col2 = st.columns(2)
        
        with col1:
            stock_symbol = st.text_input("Enter stock symbol", value="AAPL", help="e.g., AAPL, GOOGL, TSLA")
            search_query = st.text_input("🔍 Search Stocks", placeholder="Search by company name...")
            
            if search_query:
                search_results = data_collector.search_stocks(search_query)
                if search_results:
                    selected_stock = st.selectbox(
                        "Select from search results:",
                        options=search_results,
                        format_func=lambda x: f"{x['name']} ({x['symbol']}) - {x['market']}"
                    )
                    if selected_stock:
                        stock_symbol = selected_stock['symbol']
        
        with col2:
            time_period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
            
        if st.button("📈 Fetch Global Stock Data"):
            try:
                with st.spinner("Fetching stock data..."):
                    stock_data = data_collector.get_stock_data(stock_symbol, time_period)
                    st.session_state.stock_data = stock_data
                    st.success(f"✅ {stock_symbol} data fetched with VCP analysis!")
                    
                    # Display enhanced metrics
                    if stock_data is not None:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        current_price = stock_data['Close'].iloc[-1]
                        daily_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                        daily_change_pct = (daily_change / stock_data['Close'].iloc[-2]) * 100
                        current_trend = stock_data['Trend'].iloc[-1] if 'Trend' in stock_data.columns else 'N/A'
                        vcp_signal = stock_data['VCP_Signal'].iloc[-1] if 'VCP_Signal' in stock_data.columns else False
                        
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("Daily Change", f"{daily_change:.2f} ({daily_change_pct:+.1f}%)")
                        with col3:
                            st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,}")
                        with col4:
                            trend_color = "🟢" if current_trend == "Bullish" else "🔴" if current_trend == "Bearish" else "🟡"
                            st.metric("Trend", f"{trend_color} {current_trend}")
                        with col5:
                            vcp_status = "✅ VCP Active" if vcp_signal else "❌ No VCP"
                            st.metric("VCP Pattern", vcp_status)
                    
                    # Enhanced visualization with trend signals
                    fig = go.Figure()
                    
                    # Main price line
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add moving averages if available
                    if 'MA_20' in stock_data.columns:
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['MA_20'],
                            mode='lines',
                            name='MA 20',
                            line=dict(color='orange', width=1)
                        ))
                    
                    if 'MA_50' in stock_data.columns:
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['MA_50'],
                            mode='lines',
                            name='MA 50',
                            line=dict(color='red', width=1)
                        ))
                    
                    # Add trend signals
                    if 'Golden_Cross' in stock_data.columns:
                        golden_cross_points = stock_data[stock_data['Golden_Cross']]
                        if not golden_cross_points.empty:
                            fig.add_trace(go.Scatter(
                                x=golden_cross_points.index,
                                y=golden_cross_points['Close'],
                                mode='markers',
                                name='🟢 Bullish Signal',
                                marker=dict(color='green', size=10, symbol='triangle-up')
                            ))
                    
                    if 'Death_Cross' in stock_data.columns:
                        death_cross_points = stock_data[stock_data['Death_Cross']]
                        if not death_cross_points.empty:
                            fig.add_trace(go.Scatter(
                                x=death_cross_points.index,
                                y=death_cross_points['Close'],
                                mode='markers',
                                name='🔴 Bearish Signal',
                                marker=dict(color='red', size=10, symbol='triangle-down')
                            ))
                    
                    fig.update_layout(
                        title=f"{stock_symbol} - Technical Analysis with Trend Signals",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error fetching stock data: {str(e)}")
    
    with market_tab2:
        st.markdown("#### Indian Stock Markets (NSE)")
        col1, col2 = st.columns(2)
        
        with col1:
            indian_stock = st.selectbox(
                "Select Indian Stock",
                options=list(data_collector.indian_stocks.keys()),
                help="Popular Indian stocks from NSE"
            )
            indian_search = st.text_input("🔍 Search Indian Stocks", placeholder="Search Indian companies...")
            
            if indian_search:
                indian_results = [stock for stock in data_collector.indian_stocks.keys() 
                                if indian_search.upper() in stock]
                if indian_results:
                    indian_stock = st.selectbox("Search Results:", indian_results)
        
        with col2:
            time_period_indian = st.selectbox("Time Period ", ["1mo", "3mo", "6mo", "1y", "2y"], index=3, key="indian_period")
        
        if st.button("📈 Fetch Indian Stock Data"):
            try:
                indian_symbol = data_collector.indian_stocks.get(indian_stock, f"{indian_stock}.NS")
                with st.spinner("Fetching Indian stock data..."):
                    stock_data = data_collector.get_stock_data(indian_symbol, time_period_indian)
                    st.session_state.stock_data = stock_data
                    st.success(f"✅ {indian_stock} data fetched with VCP analysis!")
                    
                    # Display enhanced metrics for Indian stocks
                    if stock_data is not None:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        current_price = stock_data['Close'].iloc[-1]
                        daily_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                        daily_change_pct = (daily_change / stock_data['Close'].iloc[-2]) * 100
                        current_trend = stock_data['Trend'].iloc[-1] if 'Trend' in stock_data.columns else 'N/A'
                        vcp_signal = stock_data['VCP_Signal'].iloc[-1] if 'VCP_Signal' in stock_data.columns else False
                        
                        with col1:
                            st.metric("Current Price", f"₹{current_price:.2f}")
                        with col2:
                            st.metric("Daily Change", f"₹{daily_change:.2f} ({daily_change_pct:+.1f}%)")
                        with col3:
                            st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,}")
                        with col4:
                            trend_color = "🟢" if current_trend == "Bullish" else "🔴" if current_trend == "Bearish" else "🟡"
                            st.metric("Trend", f"{trend_color} {current_trend}")
                        with col5:
                            vcp_status = "✅ VCP Active" if vcp_signal else "❌ No VCP"
                            st.metric("VCP Pattern", vcp_status)
                    
                    # Same enhanced visualization for Indian stocks
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    if 'MA_20' in stock_data.columns:
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['MA_20'],
                            mode='lines',
                            name='MA 20',
                            line=dict(color='orange', width=1)
                        ))
                    
                    if 'MA_50' in stock_data.columns:
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['MA_50'],
                            mode='lines',
                            name='MA 50',
                            line=dict(color='red', width=1)
                        ))
                    
                    fig.update_layout(
                        title=f"{indian_stock} ({indian_symbol}) - Technical Analysis",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error fetching Indian stock data: {str(e)}")
    
    # Enhanced Economic Indicators Section
    st.subheader("🏛️ Enhanced Economic Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_country = st.selectbox(
            "🌍 Select Country",
            options=['US', 'India', 'UK', 'Germany', 'Japan'],
            help="Choose country for economic indicators"
        )
    
    with col2:
        indicator_search = st.text_input("🔍 Search Indicators", placeholder="Search economic indicators...")
    
    if st.button("📊 Load Enhanced Economic Data"):
        try:
            with st.spinner("Loading economic indicators..."):
                economic_data = data_collector.get_economic_indicators(selected_country)
                st.session_state.economic_data = economic_data
                st.success(f"✅ Economic data for {selected_country} loaded!")
                
                # Show key metrics based on country
                if not economic_data.empty:
                    metrics = economic_data.iloc[-1]
                    cols = st.columns(len(economic_data.columns))
                    
                    for i, col_name in enumerate(economic_data.columns):
                        with cols[i]:
                            value = metrics.get(col_name, 0)
                            st.metric(
                                col_name.replace('_', ' ').title(),
                                f"{value:.2f}%"
                            )
                    
                    # Enhanced economic indicators chart
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=[col.replace('_', ' ').title() for col in economic_data.columns[:4]]
                    )
                    
                    for i, col in enumerate(economic_data.columns[:4]):
                        row = (i // 2) + 1
                        col_pos = (i % 2) + 1
                        
                        fig.add_trace(
                            go.Scatter(
                                x=economic_data.index,
                                y=economic_data[col],
                                name=col.replace('_', ' ').title(),
                                mode='lines+markers'
                            ),
                            row=row, col=col_pos
                        )
                    
                    fig.update_layout(
                        height=600,
                        showlegend=False,
                        title=f"Economic Indicators - {selected_country}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error loading economic data: {str(e)}")

# Market Forecasting page
elif page == "📈 Market Forecasting":
    st.header("📈 Market Forecasting & Trend Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first!")
        st.stop()
    
    # Get available data
    data_source = st.selectbox(
        "Select data source:",
        ["Sample Data", "Uploaded Data", "Stock Data"]
    )
    
    if data_source == "Sample Data" and 'sample_data' in st.session_state:
        df = st.session_state.sample_data
    elif data_source == "Uploaded Data" and 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
    elif data_source == "Stock Data" and 'stock_data' in st.session_state:
        df = st.session_state.stock_data
    else:
        st.error("Selected data source not available!")
        st.stop()
    
    # Preprocessing
    st.subheader("🔧 Data Preprocessing")
    with st.expander("Data Preparation Steps"):
        processed_data = preprocessor.prepare_time_series_data(df)
        st.success("✅ Data preprocessing completed")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Total Records", len(processed_data))
        with col_info2:
            st.metric("Date Range", f"{processed_data.index.min().strftime('%Y-%m-%d')} to {processed_data.index.max().strftime('%Y-%m-%d')}")
        with col_info3:
            st.metric("Missing Values", processed_data.isnull().sum().sum())
    
    # Model selection and parameters
    st.subheader("🤖 Forecasting Configuration")
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        model_type = st.selectbox(
            "Select forecasting model:",
            ["Prophet", "ARIMA", "Linear Regression", "Random Forest"]
        )
        
        forecast_periods = st.slider("Forecast periods (days)", 30, 365, 90)
        
    with col_model2:
        target_column = st.selectbox(
            "Select target variable:",
            processed_data.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        include_seasonality = st.checkbox("Include seasonality", value=True)
    
    # Run forecasting
    if st.button("🚀 Run Forecasting Analysis", type="primary"):
        try:
            with st.spinner("Training forecasting model..."):
                # Prepare data for forecasting
                y = processed_data[target_column]
                
                # Generate forecast
                if model_type == "Prophet":
                    forecast_result = forecaster.prophet_forecast(y, forecast_periods, include_seasonality)
                elif model_type == "ARIMA":
                    forecast_result = forecaster.arima_forecast(y, forecast_periods)
                elif model_type == "Linear Regression":
                    forecast_result = forecaster.linear_forecast(y, forecast_periods)
                else:  # Random Forest
                    forecast_result = forecaster.random_forest_forecast(y, forecast_periods)
                
                st.session_state.forecast_result = forecast_result
                st.success("✅ Forecasting completed!")
                
                # Display results
                st.subheader("📊 Forecasting Results")
                
                # Metrics
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                
                with col_metric1:
                    st.metric("Model Type", model_type)
                with col_metric2:
                    st.metric("MAPE", f"{forecast_result['metrics']['mape']:.2f}%")
                with col_metric3:
                    st.metric("RMSE", f"{forecast_result['metrics']['rmse']:.2f}")
                with col_metric4:
                    trend = "📈 Upward" if forecast_result['forecast'].iloc[-1] > forecast_result['forecast'].iloc[0] else "📉 Downward"
                    st.metric("Trend Direction", trend)
                
                # Visualization
                fig = create_trend_chart(
                    historical_data=y,
                    forecast_data=forecast_result['forecast'],
                    title=f"{target_column} Forecast - {model_type}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                st.subheader("💡 Key Insights")
                
                insights = forecaster.generate_insights(y, forecast_result['forecast'])
                for insight in insights:
                    st.info(f"📝 {insight}")
                
                # Download forecast
                csv_data = pd.concat([
                    pd.DataFrame({'Type': 'Historical', 'Date': y.index, 'Value': y.values}),
                    pd.DataFrame({'Type': 'Forecast', 'Date': forecast_result['forecast'].index, 'Value': forecast_result['forecast'].values})
                ])
                
                st.download_button(
                    label="📥 Download Forecast Data",
                    data=csv_data.to_csv(index=False).encode('utf-8'),
                    file_name=f"forecast_{model_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")

# Customer Segmentation page
elif page == "👥 Customer Segmentation":
    st.header("👥 Customer Segmentation Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Home page first!")
        st.stop()
    
    # Get available data
    if 'sample_data' in st.session_state:
        df = st.session_state.sample_data
    elif 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
    else:
        st.error("No customer data available!")
        st.stop()
    
    st.subheader("🔧 Segmentation Configuration")
    
    col_seg1, col_seg2 = st.columns(2)
    
    with col_seg1:
        segmentation_type = st.selectbox(
            "Segmentation approach:",
            ["RFM Analysis", "Behavioral Clustering", "Demographic Clustering"]
        )
        
        n_clusters = st.slider("Number of clusters", 2, 10, 5)
    
    with col_seg2:
        if segmentation_type == "RFM Analysis":
            st.info("📊 RFM Analysis segments customers based on:\n- Recency: How recently they purchased\n- Frequency: How often they purchase\n- Monetary: How much they spend")
        elif segmentation_type == "Behavioral Clustering":
            st.info("🎯 Behavioral clustering groups customers by:\n- Purchase patterns\n- Product preferences\n- Seasonal behavior")
        else:
            st.info("👤 Demographic clustering groups by:\n- Age groups\n- Geographic location\n- Income levels")
    
    # Feature selection
    st.subheader("🎛️ Feature Selection")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        selected_numeric = st.multiselect(
            "Select numeric features:",
            numeric_columns,
            default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
        )
    
    with col_feat2:
        selected_categorical = st.multiselect(
            "Select categorical features (optional):",
            categorical_columns,
            default=[]
        )
    
    if st.button("🎯 Run Customer Segmentation", type="primary"):
        try:
            with st.spinner("Performing customer segmentation..."):
                # Prepare features
                features = selected_numeric + selected_categorical
                
                if not features:
                    st.error("Please select at least one feature!")
                    st.stop()
                
                # Run segmentation
                if segmentation_type == "RFM Analysis":
                    result = segmentation.rfm_analysis(df, n_clusters)
                elif segmentation_type == "Behavioral Clustering":
                    result = segmentation.behavioral_clustering(df[features], n_clusters)
                else:
                    result = segmentation.demographic_clustering(df[features], n_clusters)
                
                st.session_state.segmentation_result = result
                st.success("✅ Customer segmentation completed!")
                
                # Display results
                st.subheader("📊 Segmentation Results")
                
                # Cluster distribution
                cluster_counts = result['labels'].value_counts().sort_index()
                
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    fig_pie = px.pie(
                        values=cluster_counts.values,
                        names=[f"Cluster {i}" for i in cluster_counts.index],
                        title="Customer Distribution by Cluster"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_dist2:
                    st.subheader("📈 Cluster Statistics")
                    for cluster_id, count in cluster_counts.items():
                        percentage = (count / len(result['labels'])) * 100
                        st.metric(f"Cluster {cluster_id}", f"{count} customers ({percentage:.1f}%)")
                
                # Cluster visualization
                if len(selected_numeric) >= 2:
                    fig_scatter = create_cluster_visualization(
                        df[selected_numeric[:2]], 
                        result['labels'],
                        selected_numeric[:2]
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Cluster profiles
                st.subheader("👤 Cluster Profiles")
                
                for cluster_id in sorted(result['labels'].unique()):
                    with st.expander(f"Cluster {cluster_id} Profile"):
                        cluster_data = df[result['labels'] == cluster_id]
                        
                        col_prof1, col_prof2 = st.columns(2)
                        
                        with col_prof1:
                            st.write("**Numeric Features:**")
                            for col in selected_numeric:
                                if col in cluster_data.columns:
                                    mean_val = cluster_data[col].mean()
                                    st.write(f"- {col}: {mean_val:.2f} (avg)")
                        
                        with col_prof2:
                            st.write("**Categorical Features:**")
                            for col in selected_categorical:
                                if col in cluster_data.columns:
                                    mode_val = cluster_data[col].mode().iloc[0] if not cluster_data[col].mode().empty else "N/A"
                                    st.write(f"- {col}: {mode_val} (most common)")
                
                # Business insights
                st.subheader("💡 Business Insights")
                insights = segmentation.generate_insights(result, df)
                for insight in insights:
                    st.info(f"📝 {insight}")
                
                # Download segmentation results
                result_df = df.copy()
                result_df['Cluster'] = result['labels']
                
                st.download_button(
                    label="📥 Download Segmentation Results",
                    data=result_df.to_csv(index=False),
                    file_name=f"customer_segmentation_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error in customer segmentation: {str(e)}")

# Sentiment Analysis page
elif page == "💭 Sentiment Analysis":
    st.header("💭 Market Sentiment Analysis")
    
    st.subheader("📝 Text Input for Sentiment Analysis")
    
    # Single text analysis
    col_sent1, col_sent2 = st.columns([2, 1])
    
    with col_sent1:
        text_input = st.text_area(
            "Enter text for sentiment analysis:",
            height=120,
            placeholder="Enter product reviews, social media posts, or market commentary...",
            help="Analyze sentiment of market-related text"
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if text_input.strip():
                try:
                    with st.spinner("Analyzing sentiment..."):
                        result = sentiment_analyzer.analyze_text(text_input)
                        
                        st.subheader("📊 Sentiment Analysis Results")
                        
                        col_res1, col_res2, col_res3 = st.columns(3)
                        
                        with col_res1:
                            emoji = "😊" if result['sentiment'] == "POSITIVE" else "😞" if result['sentiment'] == "NEGATIVE" else "😐"
                            st.success(f"{emoji} **{result['sentiment']}**")
                        
                        with col_res2:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        
                        with col_res3:
                            st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                        
                        # Sentiment distribution
                        if 'scores' in result:
                            st.subheader("📈 Sentiment Scores")
                            scores_df = pd.DataFrame([result['scores']])
                            fig_bar = px.bar(
                                x=list(result['scores'].keys()),
                                y=list(result['scores'].values()),
                                title="Sentiment Score Distribution"
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Explanation
                        st.subheader("💡 Analysis Explanation")
                        explanation = sentiment_analyzer.explain_sentiment(text_input, result)
                        st.info(explanation)
                        
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {str(e)}")
            else:
                st.warning("Please enter some text to analyze!")
    
    with col_sent2:
        st.subheader("📋 Quick Examples")
        examples = [
            ("Positive Review", "This product exceeded my expectations! Amazing quality and fast delivery. Highly recommend!"),
            ("Negative Review", "Terrible experience. Product arrived damaged and customer service was unhelpful."),
            ("Mixed Review", "The product is okay, nothing special. Price is fair but could be better quality."),
            ("Market Bullish", "Strong earnings report shows positive market trends. Investors are optimistic about growth."),
            ("Market Bearish", "Economic uncertainty and declining sales indicate potential market downturn ahead.")
        ]
        
        for label, example in examples:
            if st.button(label, key=f"example_{label}"):
                st.session_state.example_text = example
                st.rerun()
        
        if 'example_text' in st.session_state:
            st.text_area("Selected example:", value=st.session_state.example_text, height=100, disabled=True)
    
    # Batch sentiment analysis
    st.markdown("---")
    st.subheader("📊 Batch Sentiment Analysis")
    
    col_batch1, col_batch2 = st.columns(2)
    
    with col_batch1:
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            height=150,
            placeholder="""Great product, love it!
Poor quality, not worth the money.
Average experience, nothing special.
Outstanding customer service!
Disappointed with the delivery delay."""
        )
        
        if st.button("📊 Analyze All Texts"):
            if batch_text.strip():
                try:
                    texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
                    
                    if texts:
                        with st.spinner("Analyzing multiple texts..."):
                            batch_results = sentiment_analyzer.analyze_batch(texts)
                            
                            st.subheader("📈 Batch Analysis Results")
                            
                            # Summary statistics
                            positive_count = sum(1 for r in batch_results if r['sentiment'] == 'POSITIVE')
                            negative_count = sum(1 for r in batch_results if r['sentiment'] == 'NEGATIVE')
                            neutral_count = len(batch_results) - positive_count - negative_count
                            avg_confidence = np.mean([r['confidence'] for r in batch_results])
                            
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            
                            with col_stat1:
                                st.metric("Total Texts", len(texts))
                            with col_stat2:
                                st.metric("😊 Positive", f"{positive_count} ({positive_count/len(texts)*100:.1f}%)")
                            with col_stat3:
                                st.metric("😞 Negative", f"{negative_count} ({negative_count/len(texts)*100:.1f}%)")
                            with col_stat4:
                                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                            
                            # Results table
                            results_df = pd.DataFrame([
                                {
                                    'Text': text[:50] + "..." if len(text) > 50 else text,
                                    'Sentiment': result['sentiment'],
                                    'Confidence': f"{result['confidence']:.1%}",
                                    'Full_Text': text
                                }
                                for text, result in zip(texts, batch_results)
                            ])
                            
                            st.dataframe(results_df[['Text', 'Sentiment', 'Confidence']], use_container_width=True)
                            
                            # Download results
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Sentiment Results",
                                data=csv_data,
                                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                            
                except Exception as e:
                    st.error(f"Error in batch analysis: {str(e)}")
            else:
                st.warning("Please enter some texts to analyze!")
    
    with col_batch2:
        st.info("""
        ### 💡 Sentiment Analysis Features:
        
        **🎯 Capabilities:**
        - Real-time sentiment classification
        - Confidence score assessment
        - Batch processing support
        - Market-specific language understanding
        
        **📊 Use Cases:**
        - Product review analysis
        - Social media monitoring
        - Market sentiment tracking
        - Brand perception analysis
        
        **🔍 Technical Details:**
        - Transformer-based models
        - Multi-class sentiment classification
        - Confidence intervals
        - Explainable results
        """)

# Complete Dashboard page
elif page == "📊 Complete Dashboard":
    st.header("📊 Complete Market Analysis Dashboard")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data and run analyses from other pages first!")
        st.stop()
    
    # Dashboard summary
    st.subheader("🎯 Analysis Summary")
    
    col_dash1, col_dash2, col_dash3, col_dash4 = st.columns(4)
    
    with col_dash1:
        data_status = "✅ Loaded" if st.session_state.data_loaded else "❌ Not loaded"
        st.metric("Data Status", data_status)
    
    with col_dash2:
        forecast_status = "✅ Complete" if 'forecast_result' in st.session_state else "⏳ Pending"
        st.metric("Forecasting", forecast_status)
    
    with col_dash3:
        segmentation_status = "✅ Complete" if 'segmentation_result' in st.session_state else "⏳ Pending"
        st.metric("Segmentation", segmentation_status)
    
    with col_dash4:
        sentiment_status = "✅ Available" if sentiment_analyzer else "❌ Error"
        st.metric("Sentiment Engine", sentiment_status)
    
    # Integrated visualizations
    if 'forecast_result' in st.session_state:
        st.subheader("📈 Forecasting Dashboard")
        
        forecast_data = st.session_state.forecast_result
        
        # Create combined chart
        fig = go.Figure()
        
        # Add historical data if available
        if 'sample_data' in st.session_state:
            historical = st.session_state.sample_data
            if len(historical.select_dtypes(include=[np.number]).columns) > 0:
                target_col = historical.select_dtypes(include=[np.number]).columns[0]
                fig.add_trace(go.Scatter(
                    x=historical.index if hasattr(historical.index, 'name') else range(len(historical)),
                    y=historical[target_col] if target_col in historical.columns else historical.iloc[:, 0],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast'].index,
            y=forecast_data['forecast'].values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Market Trend Forecast",
            xaxis_title="Time Period",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric("Forecast Accuracy (MAPE)", f"{forecast_data['metrics']['mape']:.2f}%")
        with col_metric2:
            trend_direction = "📈 Upward" if forecast_data['forecast'].iloc[-1] > forecast_data['forecast'].iloc[0] else "📉 Downward"
            st.metric("Trend Direction", trend_direction)
        with col_metric3:
            st.metric("Confidence Level", "High" if forecast_data['metrics']['mape'] < 10 else "Medium")
    
    if 'segmentation_result' in st.session_state:
        st.subheader("👥 Customer Segmentation Overview")
        
        segmentation_data = st.session_state.segmentation_result
        cluster_counts = segmentation_data['labels'].value_counts().sort_index()
        
        col_seg_dash1, col_seg_dash2 = st.columns(2)
        
        with col_seg_dash1:
            # Cluster distribution pie chart
            fig_pie = px.pie(
                values=cluster_counts.values,
                names=[f"Segment {i}" for i in cluster_counts.index],
                title="Customer Segmentation Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_seg_dash2:
            # Cluster size metrics
            st.markdown("**Segment Sizes:**")
            for cluster_id, count in cluster_counts.items():
                percentage = (count / len(segmentation_data['labels'])) * 100
                st.metric(f"Segment {cluster_id}", f"{count} ({percentage:.1f}%)")
    
    # Actionable insights section
    st.subheader("💡 Actionable Business Insights")
    
    insights_tabs = st.tabs(["🎯 Strategic", "📊 Operational", "💰 Financial"])
    
    with insights_tabs[0]:
        st.markdown("""
        ### Strategic Insights:
        """)
        
        if 'forecast_result' in st.session_state:
            forecast_trend = "growth" if st.session_state.forecast_result['forecast'].iloc[-1] > st.session_state.forecast_result['forecast'].iloc[0] else "decline"
            st.info(f"📈 Market forecast shows {forecast_trend} trend - consider adjusting long-term strategy accordingly")
        
        if 'segmentation_result' in st.session_state:
            n_segments = len(st.session_state.segmentation_result['labels'].unique())
            st.info(f"👥 {n_segments} distinct customer segments identified - opportunities for targeted marketing strategies")
        
        st.info("🎯 AI-driven insights enable data-driven decision making and competitive advantage")
    
    with insights_tabs[1]:
        st.markdown("""
        ### Operational Recommendations:
        """)
        
        st.success("📦 Optimize inventory levels based on demand forecasting")
        st.success("🎯 Implement segment-specific marketing campaigns")
        st.success("📊 Monitor sentiment trends for proactive issue management")
        st.success("⚡ Automate routine analysis tasks for efficiency")
    
    with insights_tabs[2]:
        st.markdown("""
        ### Financial Impact:
        """)
        
        st.info("💰 Demand forecasting can reduce inventory costs by 10-30%")
        st.info("🎯 Customer segmentation can improve marketing ROI by 15-25%")
        st.info("📈 Sentiment analysis can prevent revenue loss from negative trends")
        st.info("🔍 Overall AI implementation can increase profitability by 5-15%")
    
    # Export comprehensive report
    st.subheader("📋 Export Analysis Report")
    
    if st.button("📥 Generate Complete Report", type="primary"):
        try:
            # Compile all results into comprehensive report
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'data_status': st.session_state.data_loaded,
                'forecast_available': 'forecast_result' in st.session_state,
                'segmentation_available': 'segmentation_result' in st.session_state
            }
            
            # Add forecast data if available
            if 'forecast_result' in st.session_state:
                report_data['forecast_metrics'] = st.session_state.forecast_result['metrics']
            
            # Add segmentation data if available
            if 'segmentation_result' in st.session_state:
                report_data['segmentation_summary'] = {
                    'n_clusters': len(st.session_state.segmentation_result['labels'].unique()),
                    'cluster_distribution': st.session_state.segmentation_result['labels'].value_counts().to_dict()
                }
            
            report_json = pd.Series(report_data).to_json()
            
            st.download_button(
                label="📄 Download JSON Report",
                data=report_json,
                file_name=f"market_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.success("✅ Report generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
### 🎓 AI Market Trend Analysis System

**🔬 Built with:** Python, Streamlit, Prophet, Scikit-learn, HuggingFace Transformers  
**📊 Features:** Time series forecasting, customer segmentation, sentiment analysis  
**🎯 Purpose:** Academic research and business intelligence  
**📅 Created:** August 2025

*This system demonstrates advanced AI techniques for market analysis and business intelligence.*
""")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 System Status

**🔧 Components:**
- ✅ Data Collection
- ✅ Forecasting Engine
- ✅ Clustering Analysis
- ✅ Sentiment Analysis
- ✅ Visualization Engine

**📈 Capabilities:**
- Prophet & ARIMA forecasting
- K-Means clustering
- Transformer sentiment analysis
- Interactive dashboards
- Explainable AI with SHAP

**🎯 Academic Focus:**
- Market trend prediction
- Customer behavior analysis
- Business intelligence
- AI model evaluation
""")
