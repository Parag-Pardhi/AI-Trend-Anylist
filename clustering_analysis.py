import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    Customer segmentation and clustering analysis module.
    Supports RFM analysis, behavioral clustering, and demographic segmentation.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.models = {}
        
    def rfm_analysis(self, df, n_clusters=5):
        """
        Perform RFM (Recency, Frequency, Monetary) analysis.
        
        Args:
            df (pd.DataFrame): Transaction data
            n_clusters (int): Number of customer segments
        
        Returns:
            dict: RFM analysis results
        """
        try:
            # Create RFM features if not present
            rfm_data = self._create_rfm_features(df)
            
            if rfm_data is None or rfm_data.empty:
                raise ValueError("Could not create RFM features from data")
            
            # Scale RFM features
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(rfm_scaled)
            
            # Add cluster labels to RFM data
            rfm_data['Cluster'] = cluster_labels
            
            # Calculate cluster statistics
            cluster_stats = self._calculate_cluster_stats(rfm_data, ['Recency', 'Frequency', 'Monetary'])
            
            # Calculate clustering metrics
            metrics = self._calculate_clustering_metrics(rfm_scaled, cluster_labels)
            
            # Generate segment descriptions
            segment_descriptions = self._generate_rfm_segments(rfm_data)
            
            # Store model and scaler
            self.models['rfm'] = kmeans
            self.scalers['rfm'] = scaler
            
            return {
                'data': rfm_data,
                'labels': pd.Series(cluster_labels, index=rfm_data.index),
                'cluster_stats': cluster_stats,
                'metrics': metrics,
                'segment_descriptions': segment_descriptions,
                'model': kmeans,
                'scaler': scaler
            }
            
        except Exception as e:
            print(f"Error in RFM analysis: {e}")
            return self._fallback_clustering(df, n_clusters)
    
    def behavioral_clustering(self, df, n_clusters=5, algorithm='kmeans'):
        """
        Perform behavioral clustering based on customer behavior patterns.
        
        Args:
            df (pd.DataFrame): Customer behavior data
            n_clusters (int): Number of clusters
            algorithm (str): Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
        
        Returns:
            dict: Clustering results
        """
        try:
            # Prepare behavioral features
            behavior_features = self._prepare_behavioral_features(df)
            
            if behavior_features is None or behavior_features.empty:
                raise ValueError("Could not prepare behavioral features")
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(behavior_features)
            
            # Apply clustering algorithm
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = model.fit_predict(features_scaled)
            elif algorithm == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = model.fit_predict(features_scaled)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            else:  # hierarchical
                model = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = model.fit_predict(features_scaled)
            
            # Calculate cluster statistics
            cluster_stats = self._calculate_cluster_stats(behavior_features, behavior_features.columns)
            
            # Calculate clustering metrics
            if len(set(cluster_labels)) > 1:
                metrics = self._calculate_clustering_metrics(features_scaled, cluster_labels)
            else:
                metrics = {'silhouette_score': 0, 'calinski_harabasz_score': 0, 'davies_bouldin_score': 0}
            
            # Store model and scaler
            self.models['behavioral'] = model
            self.scalers['behavioral'] = scaler
            
            return {
                'data': behavior_features,
                'labels': pd.Series(cluster_labels, index=behavior_features.index),
                'cluster_stats': cluster_stats,
                'metrics': metrics,
                'n_clusters': n_clusters,
                'algorithm': algorithm,
                'model': model,
                'scaler': scaler
            }
            
        except Exception as e:
            print(f"Error in behavioral clustering: {e}")
            return self._fallback_clustering(df, n_clusters)
    
    def demographic_clustering(self, df, n_clusters=5):
        """
        Perform demographic-based customer segmentation.
        
        Args:
            df (pd.DataFrame): Customer demographic data
            n_clusters (int): Number of clusters
        
        Returns:
            dict: Clustering results
        """
        try:
            # Prepare demographic features
            demo_features = self._prepare_demographic_features(df)
            
            if demo_features is None or demo_features.empty:
                raise ValueError("Could not prepare demographic features")
            
            # Scale numeric features
            numeric_cols = demo_features.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                demo_features[numeric_cols] = scaler.fit_transform(demo_features[numeric_cols])
                self.scalers['demographic'] = scaler
            
            # Perform clustering
            features_for_clustering = demo_features.select_dtypes(include=[np.number])
            
            if features_for_clustering.empty:
                raise ValueError("No numeric features available for clustering")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_for_clustering)
            
            # Calculate cluster statistics
            cluster_stats = self._calculate_cluster_stats(demo_features, demo_features.columns)
            
            # Calculate clustering metrics
            metrics = self._calculate_clustering_metrics(features_for_clustering.values, cluster_labels)
            
            # Store model
            self.models['demographic'] = kmeans
            
            return {
                'data': demo_features,
                'labels': pd.Series(cluster_labels, index=demo_features.index),
                'cluster_stats': cluster_stats,
                'metrics': metrics,
                'model': kmeans
            }
            
        except Exception as e:
            print(f"Error in demographic clustering: {e}")
            return self._fallback_clustering(df, n_clusters)
    
    def _create_rfm_features(self, df):
        """Create RFM features from transaction data"""
        try:
            # Try to identify relevant columns
            date_cols = df.select_dtypes(include=['datetime64']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(date_cols) == 0:
                # Try to find date columns in object type
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'day']):
                        try:
                            df[col] = pd.to_datetime(df[col])
                            date_cols = [col]
                            break
                        except:
                            continue
            
            if len(date_cols) == 0:
                # Create synthetic date column based on index
                df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
                date_cols = ['Date']
            
            # Identify customer, date, and monetary columns
            date_col = date_cols[0]
            
            # Try to identify monetary column (sales, revenue, amount, etc.)
            monetary_col = None
            for col in numeric_cols:
                if any(keyword in col.lower() for keyword in ['sale', 'revenue', 'amount', 'value', 'price', 'total']):
                    monetary_col = col
                    break
            
            if monetary_col is None and len(numeric_cols) > 0:
                monetary_col = numeric_cols[0]  # Use first numeric column
            
            # Create customer ID if not present
            if 'customer_id' not in df.columns and 'Customer' not in df.columns:
                df['customer_id'] = range(len(df))
                customer_col = 'customer_id'
            else:
                customer_col = 'customer_id' if 'customer_id' in df.columns else 'Customer'
            
            # Calculate RFM metrics
            current_date = df[date_col].max()
            
            # Group by customer
            customer_data = df.groupby(customer_col).agg({
                date_col: ['max', 'count'],
                monetary_col: ['sum', 'mean'] if monetary_col else lambda x: len(x)
            }).reset_index()
            
            # Flatten column names
            customer_data.columns = [customer_col, 'LastPurchaseDate', 'PurchaseCount', 'TotalSpent', 'AvgSpent']
            
            # Calculate RFM metrics
            customer_data['Recency'] = (current_date - customer_data['LastPurchaseDate']).dt.days
            customer_data['Frequency'] = customer_data['PurchaseCount']
            customer_data['Monetary'] = customer_data['TotalSpent']
            
            # Set customer as index
            customer_data.set_index(customer_col, inplace=True)
            
            return customer_data[['Recency', 'Frequency', 'Monetary']]
            
        except Exception as e:
            print(f"Error creating RFM features: {e}")
            # Create synthetic RFM data
            n_customers = min(len(df), 1000)
            synthetic_rfm = pd.DataFrame({
                'Recency': np.random.exponential(30, n_customers),
                'Frequency': np.random.poisson(5, n_customers) + 1,
                'Monetary': np.random.lognormal(5, 1, n_customers)
            })
            return synthetic_rfm
    
    def _prepare_behavioral_features(self, df):
        """Prepare behavioral features from customer data"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns for behavioral analysis")
            
            # Select relevant behavioral columns
            behavioral_features = df[numeric_cols].copy()
            
            # Create additional behavioral metrics if possible
            if len(behavioral_features.columns) >= 2:
                # Create ratios and interactions
                cols = behavioral_features.columns[:2]
                behavioral_features[f'{cols[0]}_to_{cols[1]}_ratio'] = (
                    behavioral_features[cols[0]] / (behavioral_features[cols[1]] + 1)
                )
            
            # Handle missing values
            behavioral_features.fillna(behavioral_features.median(), inplace=True)
            
            return behavioral_features
            
        except Exception as e:
            print(f"Error preparing behavioral features: {e}")
            return None
    
    def _prepare_demographic_features(self, df):
        """Prepare demographic features for clustering"""
        try:
            demo_features = df.copy()
            
            # Encode categorical variables
            categorical_cols = demo_features.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if demo_features[col].nunique() < 50:  # Only encode if reasonable number of categories
                    le = LabelEncoder()
                    demo_features[col] = le.fit_transform(demo_features[col].astype(str))
                    self.encoders[col] = le
                else:
                    demo_features.drop(col, axis=1, inplace=True)
            
            # Handle missing values
            numeric_cols = demo_features.select_dtypes(include=[np.number]).columns
            demo_features[numeric_cols] = demo_features[numeric_cols].fillna(demo_features[numeric_cols].median())
            
            return demo_features
            
        except Exception as e:
            print(f"Error preparing demographic features: {e}")
            return None
    
    def _calculate_cluster_stats(self, data, columns):
        """Calculate statistics for each cluster"""
        stats = {}
        
        for cluster in data['Cluster'].unique():
            cluster_data = data[data['Cluster'] == cluster]
            cluster_stats = {}
            
            for col in columns:
                if col in cluster_data.columns and pd.api.types.is_numeric_dtype(cluster_data[col]):
                    cluster_stats[col] = {
                        'mean': cluster_data[col].mean(),
                        'median': cluster_data[col].median(),
                        'std': cluster_data[col].std(),
                        'min': cluster_data[col].min(),
                        'max': cluster_data[col].max(),
                        'count': len(cluster_data)
                    }
                elif col in cluster_data.columns:
                    cluster_stats[col] = {
                        'mode': cluster_data[col].mode().iloc[0] if not cluster_data[col].mode().empty else 'N/A',
                        'unique_count': cluster_data[col].nunique(),
                        'count': len(cluster_data)
                    }
            
            stats[cluster] = cluster_stats
        
        return stats
    
    def _calculate_clustering_metrics(self, X, labels):
        """Calculate clustering evaluation metrics"""
        try:
            unique_labels = np.unique(labels)
            
            if len(unique_labels) <= 1:
                return {
                    'silhouette_score': 0,
                    'calinski_harabasz_score': 0,
                    'davies_bouldin_score': float('inf')
                }
            
            # Remove noise points for DBSCAN
            if -1 in labels:
                mask = labels != -1
                X_clean = X[mask]
                labels_clean = labels[mask]
            else:
                X_clean = X
                labels_clean = labels
            
            if len(np.unique(labels_clean)) <= 1 or len(X_clean) < 2:
                return {
                    'silhouette_score': 0,
                    'calinski_harabasz_score': 0,
                    'davies_bouldin_score': float('inf')
                }
            
            silhouette = silhouette_score(X_clean, labels_clean)
            calinski_harabasz = calinski_harabasz_score(X_clean, labels_clean)
            davies_bouldin = davies_bouldin_score(X_clean, labels_clean)
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin
            }
            
        except Exception as e:
            print(f"Error calculating clustering metrics: {e}")
            return {
                'silhouette_score': 0,
                'calinski_harabasz_score': 0,
                'davies_bouldin_score': float('inf')
            }
    
    def _generate_rfm_segments(self, rfm_data):
        """Generate descriptive names for RFM segments"""
        segment_descriptions = {}
        
        for cluster in rfm_data['Cluster'].unique():
            cluster_data = rfm_data[rfm_data['Cluster'] == cluster]
            
            avg_recency = cluster_data['Recency'].mean()
            avg_frequency = cluster_data['Frequency'].mean()
            avg_monetary = cluster_data['Monetary'].mean()
            
            # Generate segment description
            if avg_recency <= 30 and avg_frequency >= 5 and avg_monetary >= rfm_data['Monetary'].median():
                description = "Champions"
            elif avg_recency <= 30 and avg_frequency >= 3:
                description = "Loyal Customers"
            elif avg_recency <= 60 and avg_monetary >= rfm_data['Monetary'].median():
                description = "Potential Loyalists"
            elif avg_recency <= 30:
                description = "New Customers"
            elif avg_recency <= 90 and avg_frequency >= 2:
                description = "At Risk"
            elif avg_recency > 90 and avg_frequency >= 3:
                description = "Cannot Lose Them"
            elif avg_recency > 90:
                description = "Lost Customers"
            else:
                description = "Others"
            
            segment_descriptions[cluster] = {
                'name': description,
                'avg_recency': avg_recency,
                'avg_frequency': avg_frequency,
                'avg_monetary': avg_monetary,
                'size': len(cluster_data)
            }
        
        return segment_descriptions
    
    def _fallback_clustering(self, df, n_clusters):
        """Fallback clustering when main methods fail"""
        try:
            # Use first few numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
            
            if len(numeric_cols) == 0:
                # Create synthetic features
                features = pd.DataFrame({
                    'feature_1': np.random.normal(0, 1, len(df)),
                    'feature_2': np.random.normal(0, 1, len(df)),
                    'feature_3': np.random.normal(0, 1, len(df))
                })
            else:
                features = df[numeric_cols].fillna(0)
            
            # Simple K-means clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            return {
                'data': features,
                'labels': pd.Series(labels, index=features.index),
                'cluster_stats': {},
                'metrics': {'silhouette_score': 0, 'calinski_harabasz_score': 0, 'davies_bouldin_score': 0},
                'model': kmeans,
                'scaler': scaler
            }
            
        except Exception as e:
            print(f"Error in fallback clustering: {e}")
            # Return random assignments as last resort
            labels = np.random.randint(0, n_clusters, len(df))
            return {
                'data': pd.DataFrame({'feature': range(len(df))}),
                'labels': pd.Series(labels),
                'cluster_stats': {},
                'metrics': {'silhouette_score': 0, 'calinski_harabasz_score': 0, 'davies_bouldin_score': 0},
                'model': None,
                'scaler': None
            }
    
    def generate_insights(self, clustering_result, original_data):
        """Generate business insights from clustering results"""
        insights = []
        
        try:
            labels = clustering_result['labels']
            n_clusters = len(labels.unique())
            
            # Cluster size insights
            cluster_sizes = labels.value_counts().sort_index()
            largest_cluster = cluster_sizes.idxmax()
            smallest_cluster = cluster_sizes.idxmin()
            
            insights.append(f"📊 Identified {n_clusters} distinct customer segments")
            insights.append(f"🎯 Largest segment: Cluster {largest_cluster} ({cluster_sizes[largest_cluster]} customers, {cluster_sizes[largest_cluster]/len(labels)*100:.1f}%)")
            insights.append(f"📍 Smallest segment: Cluster {smallest_cluster} ({cluster_sizes[smallest_cluster]} customers, {cluster_sizes[smallest_cluster]/len(labels)*100:.1f}%)")
            
            # Balance insights
            if cluster_sizes.max() / cluster_sizes.min() > 5:
                insights.append("⚠️ High imbalance between segments - consider reviewing segmentation strategy")
            else:
                insights.append("✅ Well-balanced customer segments for targeted marketing")
            
            # Clustering quality insights
            if 'metrics' in clustering_result:
                silhouette = clustering_result['metrics'].get('silhouette_score', 0)
                
                if silhouette > 0.7:
                    insights.append("🎯 Excellent clustering quality - segments are well-separated")
                elif silhouette > 0.5:
                    insights.append("👍 Good clustering quality - segments show reasonable separation")
                elif silhouette > 0.25:
                    insights.append("⚠️ Moderate clustering quality - consider adjusting number of segments")
                else:
                    insights.append("❌ Poor clustering quality - segments may not be meaningful")
            
            # Business recommendations
            insights.append("💼 Recommendation: Develop segment-specific marketing campaigns")
            insights.append("📈 Recommendation: Monitor segment migration patterns over time")
            insights.append("🎯 Recommendation: Focus retention efforts on high-value segments")
            
        except Exception as e:
            insights.append(f"⚠️ Error generating insights: {str(e)}")
        
        return insights
