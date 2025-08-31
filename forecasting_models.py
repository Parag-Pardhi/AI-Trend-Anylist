import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """
    Time series forecasting module supporting multiple algorithms.
    Includes Prophet, ARIMA, Linear Regression, and Random Forest models.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def prophet_forecast(self, y, periods=30, include_seasonality=True):
        """
        Perform time series forecasting using Facebook Prophet.
        
        Args:
            y (pd.Series): Time series data with datetime index
            periods (int): Number of periods to forecast
            include_seasonality (bool): Include seasonal components
        
        Returns:
            dict: Forecast results with metrics
        """
        try:
            # Prepare data for Prophet
            df_prophet = pd.DataFrame({
                'ds': y.index,
                'y': y.values
            })
            
            # Initialize Prophet model
            prophet_params = {
                'yearly_seasonality': include_seasonality,
                'weekly_seasonality': include_seasonality,
                'daily_seasonality': False,
                'seasonality_mode': 'multiplicative' if include_seasonality else 'additive'
            }
            
            model = Prophet(**prophet_params)
            
            # Add custom seasonalities if requested
            if include_seasonality:
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
            
            # Fit the model
            model.fit(df_prophet)
            
            # Create future dates
            future = model.make_future_dataframe(periods=periods)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract forecast values
            forecast_series = pd.Series(
                forecast['yhat'].iloc[-periods:].values,
                index=pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), 
                                  periods=periods, freq='D')
            )
            
            # Calculate metrics on historical data
            historical_forecast = forecast['yhat'].iloc[:-periods]
            metrics = self._calculate_metrics(y.values, historical_forecast.values)
            
            # Store model
            self.models['prophet'] = model
            
            return {
                'forecast': forecast_series,
                'model': model,
                'metrics': metrics,
                'confidence_intervals': {
                    'lower': pd.Series(forecast['yhat_lower'].iloc[-periods:].values, 
                                     index=forecast_series.index),
                    'upper': pd.Series(forecast['yhat_upper'].iloc[-periods:].values, 
                                     index=forecast_series.index)
                },
                'components': {
                    'trend': pd.Series(forecast['trend'].iloc[-periods:].values, 
                                     index=forecast_series.index),
                    'seasonal': pd.Series(forecast['yearly'].iloc[-periods:].values, 
                                        index=forecast_series.index) if include_seasonality else None
                }
            }
            
        except Exception as e:
            print(f"Error in Prophet forecasting: {e}")
            return self._fallback_forecast(y, periods)
    
    def arima_forecast(self, y, periods=30, order=None):
        """
        Perform ARIMA forecasting.
        
        Args:
            y (pd.Series): Time series data
            periods (int): Number of periods to forecast
            order (tuple): ARIMA order (p, d, q). If None, auto-determined
        
        Returns:
            dict: Forecast results with metrics
        """
        try:
            # Auto-determine ARIMA order if not provided
            if order is None:
                order = self._auto_arima_order(y)
            
            # Fit ARIMA model
            model = ARIMA(y, order=order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_result = fitted_model.forecast(steps=periods)
            
            # Create forecast series
            forecast_series = pd.Series(
                forecast_result,
                index=pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), 
                                  periods=periods, freq='D')
            )
            
            # Calculate metrics
            fitted_values = fitted_model.fittedvalues
            metrics = self._calculate_metrics(y.values[1:], fitted_values.values)  # Skip first value due to differencing
            
            # Store model
            self.models['arima'] = fitted_model
            
            return {
                'forecast': forecast_series,
                'model': fitted_model,
                'metrics': metrics,
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
        except Exception as e:
            print(f"Error in ARIMA forecasting: {e}")
            return self._fallback_forecast(y, periods)
    
    def linear_forecast(self, y, periods=30):
        """
        Simple linear regression forecast.
        
        Args:
            y (pd.Series): Time series data
            periods (int): Number of periods to forecast
        
        Returns:
            dict: Forecast results with metrics
        """
        try:
            # Prepare features (time index as numeric)
            X = np.arange(len(y)).reshape(-1, 1)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y.values)
            
            # Generate forecast
            future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
            forecast_values = model.predict(future_X)
            
            # Create forecast series
            forecast_series = pd.Series(
                forecast_values,
                index=pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), 
                                  periods=periods, freq='D')
            )
            
            # Calculate metrics
            fitted_values = model.predict(X)
            metrics = self._calculate_metrics(y.values, fitted_values)
            
            # Store model
            self.models['linear'] = model
            
            return {
                'forecast': forecast_series,
                'model': model,
                'metrics': metrics,
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r_squared': model.score(X, y.values)
            }
            
        except Exception as e:
            print(f"Error in linear forecasting: {e}")
            return self._fallback_forecast(y, periods)
    
    def random_forest_forecast(self, y, periods=30, n_estimators=100, lookback=7):
        """
        Random Forest regression forecast using lagged features.
        
        Args:
            y (pd.Series): Time series data
            periods (int): Number of periods to forecast
            n_estimators (int): Number of trees in forest
            lookback (int): Number of lagged values to use as features
        
        Returns:
            dict: Forecast results with metrics
        """
        try:
            # Create lagged features
            X, y_train = self._create_lagged_features(y, lookback)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit Random Forest
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_scaled, y_train)
            
            # Generate forecast iteratively
            forecast_values = []
            last_values = y.values[-lookback:].tolist()
            
            for _ in range(periods):
                # Prepare features for next prediction
                features = np.array(last_values[-lookback:]).reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                # Predict next value
                next_value = model.predict(features_scaled)[0]
                forecast_values.append(next_value)
                
                # Update last_values for next iteration
                last_values.append(next_value)
            
            # Create forecast series
            forecast_series = pd.Series(
                forecast_values,
                index=pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), 
                                  periods=periods, freq='D')
            )
            
            # Calculate metrics on training data
            fitted_values = model.predict(X_scaled)
            metrics = self._calculate_metrics(y_train, fitted_values)
            
            # Store model and scaler
            self.models['random_forest'] = model
            self.scalers['random_forest'] = scaler
            
            return {
                'forecast': forecast_series,
                'model': model,
                'scaler': scaler,
                'metrics': metrics,
                'feature_importance': dict(zip([f'lag_{i+1}' for i in range(lookback)], 
                                             model.feature_importances_)),
                'lookback_period': lookback
            }
            
        except Exception as e:
            print(f"Error in Random Forest forecasting: {e}")
            return self._fallback_forecast(y, periods)
    
    def _create_lagged_features(self, y, lookback):
        """Create lagged features for machine learning models"""
        X = []
        y_train = []
        
        for i in range(lookback, len(y)):
            X.append(y.values[i-lookback:i])
            y_train.append(y.values[i])
        
        return np.array(X), np.array(y_train)
    
    def _auto_arima_order(self, y, max_p=3, max_d=2, max_q=3):
        """Automatically determine ARIMA order using AIC"""
        best_aic = np.inf
        best_order = (0, 1, 0)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(y, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate forecast accuracy metrics"""
        
        # Handle cases where arrays might have different lengths
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[-min_length:]
        y_pred = y_pred[-min_length:]
        
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            
            # Additional metrics
            mean_actual = np.mean(y_true)
            mean_forecast = np.mean(y_pred)
            bias = mean_forecast - mean_actual
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape * 100,  # Convert to percentage
                'bias': bias,
                'mean_actual': mean_actual,
                'mean_forecast': mean_forecast
            }
        except:
            # Return default metrics if calculation fails
            return {
                'mae': 0.0,
                'mse': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'bias': 0.0,
                'mean_actual': np.mean(y_true),
                'mean_forecast': np.mean(y_pred)
            }
    
    def _fallback_forecast(self, y, periods):
        """Fallback forecast using simple trend extrapolation"""
        try:
            # Calculate simple linear trend
            x = np.arange(len(y))
            trend = np.polyfit(x, y.values, 1)
            
            # Extrapolate trend
            future_x = np.arange(len(y), len(y) + periods)
            forecast_values = np.polyval(trend, future_x)
            
            # Create forecast series
            forecast_series = pd.Series(
                forecast_values,
                index=pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), 
                                  periods=periods, freq='D')
            )
            
            # Simple metrics
            fitted_values = np.polyval(trend, x)
            metrics = self._calculate_metrics(y.values, fitted_values)
            
            return {
                'forecast': forecast_series,
                'model': 'fallback_trend',
                'metrics': metrics,
                'trend_slope': trend[0],
                'trend_intercept': trend[1]
            }
        except:
            # Last resort: use mean value
            mean_value = y.mean()
            forecast_series = pd.Series(
                [mean_value] * periods,
                index=pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), 
                                  periods=periods, freq='D')
            )
            
            return {
                'forecast': forecast_series,
                'model': 'fallback_mean',
                'metrics': {'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'bias': 0,
                           'mean_actual': mean_value, 'mean_forecast': mean_value}
            }
    
    def decompose_time_series(self, y, model='additive', period=None):
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            y (pd.Series): Time series data
            model (str): 'additive' or 'multiplicative'
            period (int): Seasonal period
        
        Returns:
            dict: Decomposition components
        """
        try:
            if period is None:
                period = min(len(y) // 2, 365)  # Default to yearly or half the data length
            
            decomposition = seasonal_decompose(y, model=model, period=period)
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
        except Exception as e:
            print(f"Error in time series decomposition: {e}")
            return None
    
    def generate_insights(self, historical_data, forecast_data):
        """
        Generate business insights from forecasting results.
        
        Args:
            historical_data (pd.Series): Historical time series
            forecast_data (pd.Series): Forecast time series
        
        Returns:
            list: List of insight strings
        """
        insights = []
        
        try:
            # Trend analysis
            historical_mean = historical_data.mean()
            forecast_mean = forecast_data.mean()
            
            if forecast_mean > historical_mean * 1.05:
                insights.append(f"📈 Forecast shows positive growth trend: {((forecast_mean / historical_mean - 1) * 100):.1f}% increase expected")
            elif forecast_mean < historical_mean * 0.95:
                insights.append(f"📉 Forecast indicates declining trend: {((1 - forecast_mean / historical_mean) * 100):.1f}% decrease expected")
            else:
                insights.append("📊 Forecast suggests stable market conditions with minimal change")
            
            # Volatility analysis
            historical_volatility = historical_data.std()
            forecast_volatility = forecast_data.std()
            
            if forecast_volatility > historical_volatility * 1.2:
                insights.append("⚠️ Higher volatility expected in forecast period - consider risk management strategies")
            elif forecast_volatility < historical_volatility * 0.8:
                insights.append("✅ Lower volatility expected - more predictable market conditions ahead")
            
            # Peak and trough analysis
            forecast_max = forecast_data.max()
            forecast_min = forecast_data.min()
            historical_max = historical_data.max()
            
            if forecast_max > historical_max:
                insights.append(f"🎯 New peak expected: forecast maximum ({forecast_max:.2f}) exceeds historical high ({historical_max:.2f})")
            
            # Seasonal patterns
            if len(forecast_data) >= 30:
                monthly_avg = forecast_data.groupby(forecast_data.index.month).mean()
                best_month = monthly_avg.idxmax()
                worst_month = monthly_avg.idxmin()
                
                month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                              7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                
                insights.append(f"📅 Best forecasted month: {month_names.get(best_month, best_month)} | Worst: {month_names.get(worst_month, worst_month)}")
            
            # Business recommendations
            trend_slope = (forecast_data.iloc[-1] - forecast_data.iloc[0]) / len(forecast_data)
            
            if trend_slope > 0:
                insights.append("💼 Recommendation: Consider increasing inventory and marketing spend to capture growth opportunity")
            else:
                insights.append("💼 Recommendation: Focus on cost optimization and customer retention strategies")
            
        except Exception as e:
            insights.append(f"⚠️ Error generating detailed insights: {str(e)}")
        
        return insights
