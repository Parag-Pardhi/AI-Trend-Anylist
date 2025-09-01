import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_trend_chart(historical_data, forecast_data, title="Market Trend Analysis", confidence_intervals=None):
    """
    Create an interactive trend chart showing historical data and forecasts.
    
    Args:
        historical_data (pd.Series): Historical time series data
        forecast_data (pd.Series): Forecast time series data
        title (str): Chart title
        confidence_intervals (dict): Optional confidence intervals with 'lower' and 'upper' keys
    
    Returns:
        plotly.graph_objects.Figure: Interactive trend chart
    """
    try:
        fig = go.Figure()
        
        # Add historical data
        if historical_data is not None and not historical_data.empty:
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data.values,
                mode='lines',
                name='Historical Data',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
            ))
        
        # Add forecast data
        if forecast_data is not None and not forecast_data.empty:
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data.values,
                mode='lines',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:.2f}<extra></extra>'
            ))
        
        # Add confidence intervals if provided
        if confidence_intervals and 'lower' in confidence_intervals and 'upper' in confidence_intervals:
            # Upper bound
            fig.add_trace(go.Scatter(
                x=confidence_intervals['upper'].index,
                y=confidence_intervals['upper'].values,
                mode='lines',
                name='Upper Confidence',
                line=dict(color='rgba(255, 127, 14, 0.3)', width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=confidence_intervals['lower'].index,
                y=confidence_intervals['lower'].values,
                mode='lines',
                name='Confidence Interval',
                line=dict(color='rgba(255, 127, 14, 0.3)', width=0),
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                hovertemplate='<b>Date:</b> %{x}<br><b>Lower:</b> %{y:.2f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=500
        )
        
        # Add annotations for forecast start
        if historical_data is not None and forecast_data is not None and not historical_data.empty and not forecast_data.empty:
            forecast_start = forecast_data.index[0]
            fig.add_vline(
                x=forecast_start,
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Start",
                annotation_position="top"
            )
        
        return fig
        
    except Exception as e:
        print(f"Error creating trend chart: {e}")
        # Return empty figure as fallback
        return go.Figure().add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

def create_cluster_visualization(data, labels, feature_names=None, title="Customer Segmentation"):
    """
    Create a scatter plot visualization for clustering results.
    
    Args:
        data (pd.DataFrame): Feature data used for clustering
        labels (pd.Series): Cluster labels
        feature_names (list): Names of features to plot (uses first 2 if not specified)
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Interactive cluster visualization
    """
    try:
        # Ensure we have at least 2 features
        if data.shape[1] < 2:
            # Add a synthetic second dimension
            data = data.copy()
            data['synthetic_feature'] = np.random.normal(0, 1, len(data))
        
        # Select features to plot
        if feature_names is None or len(feature_names) < 2:
            feature_names = data.columns[:2].tolist()
        
        x_feature = feature_names[0]
        y_feature = feature_names[1]
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'x': data[x_feature],
            'y': data[y_feature],
            'cluster': labels.astype(str)
        })
        
        # Create scatter plot
        fig = px.scatter(
            plot_data,
            x='x',
            y='y',
            color='cluster',
            title=title,
            labels={'x': x_feature, 'y': y_feature, 'cluster': 'Cluster'},
            hover_data={'cluster': True}
        )
        
        # Update traces for better styling
        fig.update_traces(
            marker=dict(size=8, opacity=0.7),
            hovertemplate=f'<b>{x_feature}:</b> %{{x:.2f}}<br><b>{y_feature}:</b> %{{y:.2f}}<br><b>Cluster:</b> %{{customdata[0]}}<extra></extra>'
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            template='plotly_white',
            height=500,
            legend=dict(
                title="Cluster",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating cluster visualization: {e}")
        # Return empty figure as fallback
        return go.Figure().add_annotation(
            text=f"Error creating cluster visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

def create_sentiment_distribution_chart(sentiment_data, title="Sentiment Distribution"):
    """
    Create a pie chart showing sentiment distribution.
    
    Args:
        sentiment_data (pd.Series or dict): Sentiment counts or distribution
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Sentiment distribution pie chart
    """
    try:
        if isinstance(sentiment_data, dict):
            labels = list(sentiment_data.keys())
            values = list(sentiment_data.values())
        else:
            sentiment_counts = sentiment_data.value_counts()
            labels = sentiment_counts.index.tolist()
            values = sentiment_counts.values.tolist()
        
        # Define colors for sentiments
        color_map = {
            'POSITIVE': '#2E8B57',  # Sea Green
            'NEGATIVE': '#DC143C',  # Crimson
            'NEUTRAL': '#FFD700',   # Gold
            'BULLISH': '#2E8B57',   # Sea Green
            'BEARISH': '#DC143C',   # Crimson
            'MIXED': '#FFD700'      # Gold
        }
        
        colors = [color_map.get(label, '#1f77b4') for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=colors),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            template='plotly_white',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating sentiment distribution chart: {e}")
        return go.Figure().add_annotation(
            text=f"Error creating sentiment chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

def create_time_series_decomposition_chart(decomposition_data, title="Time Series Decomposition"):
    """
    Create a multi-panel chart showing time series decomposition components.
    
    Args:
        decomposition_data (dict): Dictionary with 'trend', 'seasonal', 'residual', 'observed' components
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Time series decomposition chart
    """
    try:
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        components = ['observed', 'trend', 'seasonal', 'residual']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (component, color) in enumerate(zip(components, colors), 1):
            if component in decomposition_data and decomposition_data[component] is not None:
                data = decomposition_data[component]
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data.values,
                        mode='lines',
                        name=component.title(),
                        line=dict(color=color, width=1.5),
                        showlegend=False
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=800,
            template='plotly_white'
        )
        
        # Update x-axis for bottom subplot only
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        return fig
        
    except Exception as e:
        print(f"Error creating decomposition chart: {e}")
        return go.Figure().add_annotation(
            text=f"Error creating decomposition chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

def create_feature_importance_chart(importance_data, title="Feature Importance"):
    """
    Create a horizontal bar chart showing feature importance.
    
    Args:
        importance_data (dict): Dictionary with feature names as keys and importance scores as values
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Feature importance bar chart
    """
    try:
        features = list(importance_data.keys())
        importances = list(importance_data.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        features_sorted = [x[0] for x in sorted_data]
        importances_sorted = [x[1] for x in sorted_data]
        
        fig = go.Figure(data=[
            go.Bar(
                x=importances_sorted,
                y=features_sorted,
                orientation='h',
                marker=dict(color='#1f77b4'),
                hovertemplate='<b>Feature:</b> %{y}<br><b>Importance:</b> %{x:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="Importance Score",
            yaxis_title="Features",
            template='plotly_white',
            height=max(400, len(features) * 30)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating feature importance chart: {e}")
        return go.Figure().add_annotation(
            text=f"Error creating feature importance chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

def create_correlation_heatmap(data, title="Feature Correlation Heatmap"):
    """
    Create a correlation heatmap for numerical features.
    
    Args:
        data (pd.DataFrame): Data for correlation analysis
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    try:
        # Calculate correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numerical data available for correlation analysis")
        
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            template='plotly_white',
            height=500,
            width=500
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
        return go.Figure().add_annotation(
            text=f"Error creating correlation heatmap: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

def create_model_performance_chart(metrics_data, title="Model Performance Comparison"):
    """
    Create a radar chart comparing model performance metrics.
    
    Args:
        metrics_data (dict): Dictionary with model names as keys and metrics as values
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Model performance radar chart
    """
    try:
        fig = go.Figure()
        
        # Define metrics to display (normalize to 0-1 scale)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for model_name, metrics in metrics_data.items():
            # Extract and normalize metrics
            values = []
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in metrics:
                    values.append(metrics[metric])
                else:
                    values.append(0)
            
            # Add the first value at the end to close the radar chart
            values.append(values[0])
            metric_names_closed = metric_names + [metric_names[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names_closed,
                fill='toself',
                name=model_name,
                hovertemplate='<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            template='plotly_white',
            height=500
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating model performance chart: {e}")
        return go.Figure().add_annotation(
            text=f"Error creating performance chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

def create_business_metrics_dashboard(metrics_data, title="Business Impact Metrics"):
    """
    Create a comprehensive dashboard showing key business metrics.
    
    Args:
        metrics_data (dict): Dictionary with various business metrics
        title (str): Dashboard title
    
    Returns:
        plotly.graph_objects.Figure: Business metrics dashboard
    """
    try:
        # Create subplots for different metric types
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Impact', 'Customer Segments', 'Forecast Accuracy', 'Market Sentiment'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Revenue impact (if available)
        if 'revenue_data' in metrics_data:
            revenue_data = metrics_data['revenue_data']
            fig.add_trace(
                go.Bar(x=list(revenue_data.keys()), y=list(revenue_data.values()), name="Revenue"),
                row=1, col=1
            )
        
        # Customer segments (if available)
        if 'segment_data' in metrics_data:
            segment_data = metrics_data['segment_data']
            fig.add_trace(
                go.Pie(labels=list(segment_data.keys()), values=list(segment_data.values()), name="Segments"),
                row=1, col=2
            )
        
        # Forecast accuracy over time (if available)
        if 'accuracy_timeline' in metrics_data:
            accuracy_data = metrics_data['accuracy_timeline']
            fig.add_trace(
                go.Scatter(x=accuracy_data['dates'], y=accuracy_data['accuracy'], mode='lines+markers', name="Accuracy"),
                row=2, col=1
            )
        
        # Market sentiment indicator (if available)
        if 'sentiment_score' in metrics_data:
            sentiment_score = metrics_data['sentiment_score']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=sentiment_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Sentiment Score"},
                    gauge={'axis': {'range': [-1, 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [-1, -0.5], 'color': "lightgray"},
                                     {'range': [-0.5, 0.5], 'color': "gray"},
                                     {'range': [0.5, 1], 'color': "lightgreen"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                         'thickness': 0.75, 'value': 0.9}}
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating business metrics dashboard: {e}")
        return go.Figure().add_annotation(
            text=f"Error creating business dashboard: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
