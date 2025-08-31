import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import time
from textblob import TextBlob
import os
import warnings
warnings.filterwarnings('ignore')

class MarketSentimentAnalyzer:
    """
    Market sentiment analysis module using transformer-based models.
    Supports single text analysis, batch processing, and market-specific sentiment insights.
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name (str): HuggingFace model name for sentiment analysis
        """
        try:
            # Load pre-trained sentiment analysis model
            self.analyzer = pipeline(
                "sentiment-analysis", 
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True
            )
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        except Exception as e:
            print(f"Error loading primary model {model_name}: {e}")
            # Fallback to DistilBERT
            try:
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True
                )
                self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            except Exception as fallback_error:
                print(f"Error loading fallback model: {fallback_error}")
                self.analyzer = None
                self.model_name = "textblob_fallback"
                self.tokenizer = None
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Input text for sentiment analysis
        
        Returns:
            dict: Sentiment analysis results
        """
        start_time = time.time()
        
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            if self.analyzer:
                # Use transformer model
                results = self.analyzer(cleaned_text)
                
                # Extract sentiment and confidence
                if isinstance(results[0], list):
                    # Multiple scores returned
                    scores = {result['label']: result['score'] for result in results[0]}
                    best_sentiment = max(scores.keys(), key=lambda k: scores[k])
                    confidence = scores[best_sentiment]
                else:
                    # Single result
                    best_sentiment = results[0]['label']
                    confidence = results[0]['score']
                    scores = {best_sentiment: confidence}
                
                # Normalize sentiment labels
                sentiment = self._normalize_sentiment_label(best_sentiment)
                
            else:
                # Fallback to TextBlob
                blob = TextBlob(cleaned_text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment = "POSITIVE"
                    confidence = (polarity + 1) / 2
                elif polarity < -0.1:
                    sentiment = "NEGATIVE"
                    confidence = (1 - polarity) / 2
                else:
                    sentiment = "NEUTRAL"
                    confidence = 1 - abs(polarity)
                
                scores = {sentiment: confidence}
            
            processing_time = time.time() - start_time
            
            return {
                'text': text,
                'cleaned_text': cleaned_text,
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': scores,
                'processing_time': processing_time,
                'model_used': self.model_name
            }
            
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {
                'text': text,
                'sentiment': 'NEUTRAL',
                'confidence': 0.5,
                'scores': {'NEUTRAL': 0.5},
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def analyze_batch(self, texts, batch_size=32):
        """
        Analyze sentiment of multiple texts in batches.
        
        Args:
            texts (list): List of texts to analyze
            batch_size (int): Number of texts to process in each batch
        
        Returns:
            list: List of sentiment analysis results
        """
        try:
            results = []
            
            # Process in batches for efficiency
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                if self.analyzer:
                    # Use transformer model for batch processing
                    batch_results = []
                    for text in batch:
                        result = self.analyze_text(text)
                        batch_results.append(result)
                    
                    results.extend(batch_results)
                else:
                    # Fallback batch processing
                    for text in batch:
                        result = self.analyze_text(text)
                        results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            # Return individual analysis for each text
            return [self.analyze_text(text) for text in texts]
    
    def analyze_market_sentiment(self, texts, market_context=None):
        """
        Analyze market sentiment with domain-specific adjustments.
        
        Args:
            texts (list): Market-related texts
            market_context (str): Market context ('stock', 'crypto', 'retail', etc.)
        
        Returns:
            dict: Market sentiment analysis results
        """
        try:
            # Analyze individual texts
            individual_results = self.analyze_batch(texts)
            
            # Aggregate results
            sentiments = [result['sentiment'] for result in individual_results]
            confidences = [result['confidence'] for result in individual_results]
            
            # Calculate aggregate metrics
            sentiment_counts = pd.Series(sentiments).value_counts()
            avg_confidence = np.mean(confidences)
            
            # Determine overall market sentiment
            positive_ratio = sentiment_counts.get('POSITIVE', 0) / len(sentiments) if len(sentiments) > 0 else 0
            negative_ratio = sentiment_counts.get('NEGATIVE', 0) / len(sentiments) if len(sentiments) > 0 else 0
            neutral_ratio = sentiment_counts.get('NEUTRAL', 0) / len(sentiments) if len(sentiments) > 0 else 0
            
            if positive_ratio > 0.5:
                overall_sentiment = "BULLISH"
            elif negative_ratio > 0.5:
                overall_sentiment = "BEARISH"
            else:
                overall_sentiment = "MIXED"
            
            # Market-specific insights
            insights = self._generate_market_insights(
                sentiment_counts, avg_confidence, market_context
            )
            
            return {
                'individual_results': individual_results,
                'overall_sentiment': overall_sentiment,
                'sentiment_distribution': {
                    'positive': positive_ratio,
                    'negative': negative_ratio,
                    'neutral': neutral_ratio
                },
                'average_confidence': avg_confidence,
                'total_texts_analyzed': len(texts),
                'insights': insights,
                'market_context': market_context
            }
            
        except Exception as e:
            print(f"Error in market sentiment analysis: {e}")
            return {
                'overall_sentiment': 'MIXED',
                'sentiment_distribution': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'average_confidence': 0.5,
                'error': str(e)
            }
    
    def _preprocess_text(self, text):
        """
        Preprocess text for better sentiment analysis.
        
        Args:
            text (str): Raw text
        
        Returns:
            str: Preprocessed text
        """
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags (keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Limit text length for model compatibility
        if self.tokenizer:
            # Truncate to model's max length
            max_length = self.tokenizer.model_max_length
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > max_length - 2:  # Account for special tokens
                tokens = tokens[:max_length - 2]
                text = self.tokenizer.convert_tokens_to_string(tokens)
        else:
            # Fallback truncation
            text = text[:512]
        
        return text
    
    def _normalize_sentiment_label(self, label):
        """Normalize different model sentiment labels to standard format"""
        label_upper = label.upper()
        
        # Handle different model label formats
        if label_upper in ['POSITIVE', 'POS', 'LABEL_2']:
            return 'POSITIVE'
        elif label_upper in ['NEGATIVE', 'NEG', 'LABEL_0']:
            return 'NEGATIVE'
        elif label_upper in ['NEUTRAL', 'NEU', 'LABEL_1']:
            return 'NEUTRAL'
        else:
            return label_upper
    
    def _generate_market_insights(self, sentiment_counts, avg_confidence, market_context):
        """Generate market-specific insights from sentiment analysis"""
        insights = []
        
        try:
            total_texts = sentiment_counts.sum()
            
            # Overall sentiment insights
            if sentiment_counts.get('POSITIVE', 0) > sentiment_counts.get('NEGATIVE', 0) * 2:
                insights.append("🚀 Strong positive market sentiment detected - potential buying opportunity")
            elif sentiment_counts.get('NEGATIVE', 0) > sentiment_counts.get('POSITIVE', 0) * 2:
                insights.append("📉 Strong negative sentiment - market may be oversold")
            else:
                insights.append("⚖️ Mixed market sentiment - exercise caution in trading decisions")
            
            # Confidence insights
            if avg_confidence > 0.8:
                insights.append(f"🎯 High confidence in sentiment analysis ({avg_confidence:.1%}) - reliable signals")
            elif avg_confidence < 0.6:
                insights.append(f"⚠️ Lower confidence in analysis ({avg_confidence:.1%}) - consider additional data sources")
            
            # Volume insights
            if total_texts > 100:
                insights.append(f"📊 High discussion volume ({total_texts} texts) - strong market interest")
            elif total_texts < 20:
                insights.append(f"📊 Low discussion volume ({total_texts} texts) - limited market sentiment data")
            
            # Context-specific insights
            if market_context:
                if market_context.lower() == 'stock':
                    insights.append("📈 Consider correlation with trading volume and price movements")
                elif market_context.lower() == 'crypto':
                    insights.append("₿ Crypto sentiment often leads price movements - monitor closely")
                elif market_context.lower() == 'retail':
                    insights.append("🛒 Customer sentiment directly impacts purchasing decisions")
            
            return insights
            
        except Exception as e:
            return [f"⚠️ Error generating insights: {str(e)}"]
    
    def explain_sentiment(self, text, analysis_result):
        """
        Generate an explanation for the sentiment analysis result.
        
        Args:
            text (str): Original text
            analysis_result (dict): Sentiment analysis result
        
        Returns:
            str: Human-readable explanation
        """
        try:
            sentiment = analysis_result['sentiment']
            confidence = analysis_result['confidence']
            
            # Base explanation
            explanation = f"The AI model classified this text as **{sentiment}** with {confidence:.1%} confidence. "
            
            # Confidence level explanation
            if confidence > 0.9:
                explanation += "This is a very high confidence prediction, indicating clear sentiment indicators in the text. "
            elif confidence > 0.7:
                explanation += "This is a high confidence prediction with strong sentiment signals. "
            elif confidence > 0.5:
                explanation += "This is a moderate confidence prediction - the sentiment is somewhat ambiguous. "
            else:
                explanation += "This is a low confidence prediction - the text contains mixed or weak sentiment signals. "
            
            # Identify potential sentiment indicators
            text_lower = text.lower()
            
            positive_words = ['great', 'excellent', 'amazing', 'love', 'good', 'best', 'outstanding', 'fantastic', 'wonderful']
            negative_words = ['terrible', 'awful', 'hate', 'worst', 'bad', 'horrible', 'disappointing', 'poor']
            
            found_positive = [word for word in positive_words if word in text_lower]
            found_negative = [word for word in negative_words if word in text_lower]
            
            if found_positive and sentiment == 'POSITIVE':
                explanation += f"Positive indicators detected: {', '.join(found_positive[:3])}. "
            elif found_negative and sentiment == 'NEGATIVE':
                explanation += f"Negative indicators detected: {', '.join(found_negative[:3])}. "
            
            # Model information
            explanation += f"Analysis performed using {self.model_name} model."
            
            return explanation
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def get_sentiment_trends(self, texts_with_dates):
        """
        Analyze sentiment trends over time.
        
        Args:
            texts_with_dates (list): List of tuples (date, text)
        
        Returns:
            pd.DataFrame: Sentiment trends over time
        """
        try:
            # Analyze all texts
            texts = [text for date, text in texts_with_dates]
            dates = [date for date, text in texts_with_dates]
            
            results = self.analyze_batch(texts)
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'text': texts,
                'sentiment': [r['sentiment'] for r in results],
                'confidence': [r['confidence'] for r in results]
            })
            
            # Aggregate by date
            daily_sentiment = df.groupby('date').agg({
                'sentiment': lambda x: x.mode().iloc[0] if not x.mode().empty else 'NEUTRAL',
                'confidence': 'mean'
            }).reset_index()
            
            # Calculate sentiment scores
            sentiment_mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
            daily_sentiment['sentiment_score'] = daily_sentiment['sentiment'].apply(lambda x: sentiment_mapping.get(x, 0))
            
            return daily_sentiment
            
        except Exception as e:
            print(f"Error analyzing sentiment trends: {e}")
            return pd.DataFrame()
    
    def benchmark_model_performance(self, test_texts_with_labels):
        """
        Benchmark the sentiment model performance against labeled data.
        
        Args:
            test_texts_with_labels (list): List of tuples (text, true_label)
        
        Returns:
            dict: Performance metrics
        """
        try:
            texts = [text for text, label in test_texts_with_labels]
            true_labels = [label for text, label in test_texts_with_labels]
            
            # Analyze all texts
            results = self.analyze_batch(texts)
            predicted_labels = [r['sentiment'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            # Calculate accuracy
            correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
            accuracy = correct_predictions / len(true_labels)
            
            # Calculate per-class metrics
            unique_labels = list(set(true_labels))
            class_metrics = {}
            
            for label in unique_labels:
                true_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) 
                                   if true == label and pred == label)
                false_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) 
                                    if true != label and pred == label)
                false_negatives = sum(1 for true, pred in zip(true_labels, predicted_labels) 
                                    if true == label and pred != label)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }
            
            return {
                'overall_accuracy': accuracy,
                'average_confidence': np.mean(confidences),
                'class_metrics': class_metrics,
                'total_samples': len(test_texts_with_labels),
                'model_name': self.model_name
            }
            
        except Exception as e:
            print(f"Error benchmarking model: {e}")
            return {'error': str(e)}
