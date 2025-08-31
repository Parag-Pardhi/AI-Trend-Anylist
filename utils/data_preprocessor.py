import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing module for market trend analysis.
    Handles data cleaning, feature engineering, and preparation for various AI models.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_stats = {}
        
    def prepare_time_series_data(self, df, date_column=None, target_column=None, freq='D'):
        """
        Prepare data for time series analysis and forecasting.
        
        Args:
            df (pd.DataFrame): Input dataframe
            date_column (str): Name of date column, if None tries to auto-detect
            target_column (str): Target variable for forecasting
            freq (str): Frequency for time series resampling
        
        Returns:
            pd.DataFrame: Preprocessed time series data
        """
        try:
            df_processed = df.copy()
            
            # Auto-detect date column if not specified
            if date_column is None:
                date_column = self._detect_date_column(df_processed)
            
            if date_column is None:
                # Create synthetic date index
                df_processed.index = pd.date_range(
                    start='2022-01-01', 
                    periods=len(df_processed), 
                    freq=freq
                )
            else:
                # Convert date column to datetime and set as index
                df_processed[date_column] = pd.to_datetime(df_processed[date_column])
                df_processed.set_index(date_column, inplace=True)
                df_processed.sort_index(inplace=True)
            
            # Handle duplicate timestamps
            df_processed = df_processed[~df_processed.index.duplicated(keep='first')]
            
            # Resample to ensure consistent frequency
            if freq != 'infer':
                numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    df_processed = df_processed.resample(freq).agg({
                        **{col: 'mean' for col in numeric_columns},
                        **{col: 'first' for col in df_processed.columns if col not in numeric_columns}
                    })
            
            # Fill missing values in time series
            df_processed = self._handle_time_series_missing_values(df_processed)
            
            # Remove extreme outliers
            df_processed = self._handle_time_series_outliers(df_processed)
            
            # Create time-based features
            df_processed = self._create_time_features(df_processed)
            
            return df_processed
            
        except Exception as e:
            print(f"Error preparing time series data: {e}")
            return df
    
    def prepare_clustering_data(self, df, categorical_encoding='label', scaling_method='standard'):
        """
        Prepare data for clustering analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_encoding (str): Method for encoding categorical variables ('label', 'onehot')
            scaling_method (str): Scaling method ('standard', 'minmax', 'none')
        
        Returns:
            pd.DataFrame: Preprocessed data ready for clustering
        """
        try:
            df_processed = df.copy()
            
            # Handle missing values
            df_processed = self._handle_missing_values(df_processed, strategy='median')
            
            # Encode categorical variables
            df_processed = self._encode_categorical_variables(df_processed, method=categorical_encoding)
            
            # Scale numerical features
            if scaling_method != 'none':
                df_processed = self._scale_features(df_processed, method=scaling_method)
            
            # Remove highly correlated features
            df_processed = self._remove_high_correlation_features(df_processed, threshold=0.95)
            
            # Remove constant features
            df_processed = self._remove_constant_features(df_processed)
            
            return df_processed
            
        except Exception as e:
            print(f"Error preparing clustering data: {e}")
            return df
    
    def prepare_sentiment_data(self, texts, max_length=512):
        """
        Prepare text data for sentiment analysis.
        
        Args:
            texts (list or pd.Series): Text data
            max_length (int): Maximum text length
        
        Returns:
            list: Preprocessed text data
        """
        try:
            if isinstance(texts, pd.Series):
                texts = texts.tolist()
            
            processed_texts = []
            
            for text in texts:
                # Convert to string
                text = str(text)
                
                # Basic text cleaning
                cleaned_text = self._clean_text(text)
                
                # Truncate if too long
                if len(cleaned_text) > max_length:
                    cleaned_text = cleaned_text[:max_length]
                
                processed_texts.append(cleaned_text)
            
            return processed_texts
            
        except Exception as e:
            print(f"Error preparing sentiment data: {e}")
            return texts
    
    def create_features_for_forecasting(self, df, target_column, lag_features=True, rolling_features=True, 
                                      seasonal_features=True, lag_periods=[1, 7, 30], 
                                      rolling_windows=[7, 14, 30]):
        """
        Create features for machine learning-based forecasting.
        
        Args:
            df (pd.DataFrame): Time series dataframe
            target_column (str): Target variable column name
            lag_features (bool): Create lag features
            rolling_features (bool): Create rolling statistics features
            seasonal_features (bool): Create seasonal features
            lag_periods (list): List of lag periods to create
            rolling_windows (list): List of rolling window sizes
        
        Returns:
            pd.DataFrame: Feature-engineered dataframe
        """
        try:
            df_features = df.copy()
            
            if target_column not in df_features.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataframe")
            
            # Create lag features
            if lag_features:
                for lag in lag_periods:
                    df_features[f'{target_column}_lag_{lag}'] = df_features[target_column].shift(lag)
            
            # Create rolling statistics features
            if rolling_features:
                for window in rolling_windows:
                    df_features[f'{target_column}_rolling_mean_{window}'] = (
                        df_features[target_column].rolling(window=window).mean()
                    )
                    df_features[f'{target_column}_rolling_std_{window}'] = (
                        df_features[target_column].rolling(window=window).std()
                    )
                    df_features[f'{target_column}_rolling_min_{window}'] = (
                        df_features[target_column].rolling(window=window).min()
                    )
                    df_features[f'{target_column}_rolling_max_{window}'] = (
                        df_features[target_column].rolling(window=window).max()
                    )
            
            # Create seasonal features
            if seasonal_features:
                df_features = self._create_seasonal_features(df_features, target_column)
            
            # Create difference features
            df_features[f'{target_column}_diff_1'] = df_features[target_column].diff(1)
            df_features[f'{target_column}_diff_7'] = df_features[target_column].diff(7)
            
            # Create percentage change features
            df_features[f'{target_column}_pct_change_1'] = df_features[target_column].pct_change(1)
            df_features[f'{target_column}_pct_change_7'] = df_features[target_column].pct_change(7)
            
            # Drop rows with NaN values created by feature engineering
            df_features.dropna(inplace=True)
            
            return df_features
            
        except Exception as e:
            print(f"Error creating forecasting features: {e}")
            return df
    
    def _detect_date_column(self, df):
        """Auto-detect date column in dataframe"""
        try:
            # Check for common date column names
            date_keywords = ['date', 'time', 'timestamp', 'day', 'created', 'updated']
            
            for col in df.columns:
                if any(keyword in col.lower() for keyword in date_keywords):
                    try:
                        pd.to_datetime(df[col].head())
                        return col
                    except:
                        continue
            
            # Check for datetime types
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                return datetime_cols[0]
            
            # Try to convert object columns to datetime
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    pd.to_datetime(df[col].head())
                    return col
                except:
                    continue
            
            return None
            
        except Exception as e:
            print(f"Error detecting date column: {e}")
            return None
    
    def _handle_missing_values(self, df, strategy='median'):
        """Handle missing values in dataframe"""
        try:
            df_processed = df.copy()
            
            # Separate numeric and categorical columns
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            
            # Handle numeric missing values
            if len(numeric_cols) > 0:
                if strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                else:
                    imputer = SimpleImputer(strategy=strategy)
                
                df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
                self.imputers['numeric'] = imputer
            
            # Handle categorical missing values
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_cols] = imputer.fit_transform(df_processed[categorical_cols])
                self.imputers['categorical'] = imputer
            
            return df_processed
            
        except Exception as e:
            print(f"Error handling missing values: {e}")
            return df
    
    def _handle_time_series_missing_values(self, df):
        """Handle missing values specifically for time series data"""
        try:
            df_processed = df.copy()
            
            # Forward fill first, then backward fill
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
            
            # For remaining NaN values, use interpolation
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].interpolate(method='linear')
            
            return df_processed
            
        except Exception as e:
            print(f"Error handling time series missing values: {e}")
            return df
    
    def _handle_time_series_outliers(self, df, method='iqr', factor=1.5):
        """Handle outliers in time series data"""
        try:
            df_processed = df.copy()
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if method == 'iqr':
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    # Cap outliers instead of removing them
                    df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                
                elif method == 'zscore':
                    z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
                    df_processed = df_processed[z_scores < 3]
            
            return df_processed
            
        except Exception as e:
            print(f"Error handling outliers: {e}")
            return df
    
    def _create_time_features(self, df):
        """Create time-based features from datetime index"""
        try:
            df_processed = df.copy()
            
            # Ensure index is datetime
            if not isinstance(df_processed.index, pd.DatetimeIndex):
                return df_processed
            
            # Create time features
            dt_index = pd.to_datetime(df_processed.index)
            df_processed['year'] = dt_index.year
            df_processed['month'] = dt_index.month
            df_processed['day'] = dt_index.day
            df_processed['dayofweek'] = dt_index.dayofweek
            df_processed['dayofyear'] = dt_index.dayofyear
            df_processed['week'] = dt_index.isocalendar().week
            df_processed['quarter'] = dt_index.quarter
            df_processed['is_weekend'] = (dt_index.dayofweek >= 5).astype(int)
            df_processed['is_month_start'] = dt_index.is_month_start.astype(int)
            df_processed['is_month_end'] = dt_index.is_month_end.astype(int)
            df_processed['is_quarter_start'] = dt_index.is_quarter_start.astype(int)
            df_processed['is_quarter_end'] = dt_index.is_quarter_end.astype(int)
            
            # Create cyclical features
            df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
            df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
            df_processed['dayofweek_sin'] = np.sin(2 * np.pi * df_processed['dayofweek'] / 7)
            df_processed['dayofweek_cos'] = np.cos(2 * np.pi * df_processed['dayofweek'] / 7)
            
            return df_processed
            
        except Exception as e:
            print(f"Error creating time features: {e}")
            return df
    
    def _encode_categorical_variables(self, df, method='label'):
        """Encode categorical variables"""
        try:
            df_processed = df.copy()
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if method == 'label':
                    encoder = LabelEncoder()
                    df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
                    self.encoders[col] = encoder
                
                elif method == 'onehot':
                    # Only use one-hot encoding for columns with reasonable number of categories
                    if df_processed[col].nunique() <= 10:
                        encoded_cols = pd.get_dummies(df_processed[col], prefix=col)
                        df_processed = pd.concat([df_processed.drop(col, axis=1), encoded_cols], axis=1)
                    else:
                        # Fall back to label encoding for high cardinality
                        encoder = LabelEncoder()
                        df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
                        self.encoders[col] = encoder
            
            return df_processed
            
        except Exception as e:
            print(f"Error encoding categorical variables: {e}")
            return df
    
    def _scale_features(self, df, method='standard'):
        """Scale numerical features"""
        try:
            df_processed = df.copy()
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return df_processed
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                return df_processed
            
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            self.scalers['features'] = scaler
            
            return df_processed
            
        except Exception as e:
            print(f"Error scaling features: {e}")
            return df
    
    def _remove_high_correlation_features(self, df, threshold=0.95):
        """Remove highly correlated features"""
        try:
            df_processed = df.copy()
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return df_processed
            
            # Calculate correlation matrix
            corr_matrix = df_processed[numeric_cols].corr().abs()
            
            # Find highly correlated pairs
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
            
            # Drop highly correlated features
            df_processed = df_processed.drop(columns=to_drop)
            
            return df_processed
            
        except Exception as e:
            print(f"Error removing correlated features: {e}")
            return df
    
    def _remove_constant_features(self, df):
        """Remove features with constant values"""
        try:
            df_processed = df.copy()
            
            # Find constant features
            constant_features = [col for col in df_processed.columns if df_processed[col].nunique() <= 1]
            
            # Drop constant features
            df_processed = df_processed.drop(columns=constant_features)
            
            return df_processed
            
        except Exception as e:
            print(f"Error removing constant features: {e}")
            return df
    
    def _clean_text(self, text):
        """Clean text for sentiment analysis"""
        import re
        
        try:
            # Convert to string
            text = str(text)
            
            # Remove URLs
            text = re.sub(r'http\S+|www.\S+', '', text)
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^\w\s.,!?;:-]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return str(text)
    
    def _create_seasonal_features(self, df, target_column):
        """Create seasonal decomposition features"""
        try:
            df_processed = df.copy()
            
            # Create seasonal indicators
            df_processed['is_holiday_season'] = ((df_processed.index.month == 12) | 
                                               (df_processed.index.month == 1)).astype(int)
            df_processed['is_summer'] = ((df_processed.index.month >= 6) & 
                                       (df_processed.index.month <= 8)).astype(int)
            df_processed['is_winter'] = ((df_processed.index.month >= 12) | 
                                       (df_processed.index.month <= 2)).astype(int)
            
            # Create seasonal moving averages
            df_processed[f'{target_column}_seasonal_avg'] = (
                df_processed.groupby(df_processed.index.month)[target_column].transform('mean')
            )
            
            return df_processed
            
        except Exception as e:
            print(f"Error creating seasonal features: {e}")
            return df
    
    def get_preprocessing_summary(self):
        """Get summary of preprocessing operations performed"""
        summary = {
            'scalers_used': list(self.scalers.keys()),
            'encoders_used': list(self.encoders.keys()),
            'imputers_used': list(self.imputers.keys()),
            'feature_stats': self.feature_stats
        }
        return summary
    
    def transform_new_data(self, df, data_type='time_series'):
        """Transform new data using fitted preprocessors"""
        try:
            df_processed = df.copy()
            
            # Apply saved transformations based on data type
            if data_type == 'time_series':
                df_processed = self.prepare_time_series_data(df_processed)
            elif data_type == 'clustering':
                df_processed = self.prepare_clustering_data(df_processed)
            
            return df_processed
            
        except Exception as e:
            print(f"Error transforming new data: {e}")
            return df
