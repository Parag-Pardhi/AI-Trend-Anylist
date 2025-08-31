import streamlit as st
from transformers import pipeline
import pandas as pd
import time

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 AI Sentiment Analysis",
    page_icon="🎯",
    layout="wide"
)

# Main title
st.title("🚀 Live AI Sentiment Analysis")
st.markdown("**✨ Running live on Replit - Day 1 Success!**")
st.markdown("---")

# Load AI model with caching
@st.cache_resource
def load_sentiment_model():
    """Load the sentiment analysis model (cached for performance)"""
    print("🤖 Loading AI model...")
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize the AI model
try:
    with st.spinner("🤖 Loading AI engine... (First time takes ~30 seconds)"):
        analyzer = load_sentiment_model()
    st.success("✅ AI Engine Ready! Start analyzing sentiment below! 🎯")
    
except Exception as e:
    st.error(f"❌ Error loading AI model: {e}")
    st.info("💡 Try refreshing the page if this persists.")
    st.stop()

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Analyze Text Sentiment")
    
    # Text input
    user_text = st.text_area(
        "Enter your text to analyze:",
        height=120,
        placeholder="Example: I absolutely love this product! The quality is amazing and delivery was super fast.",
        help="Enter any text - product review, social media post, comment, feedback, etc."
    )
    
    # Analyze button
    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        if user_text.strip():
            # Show analysis in progress
            with st.spinner("🔍 Analyzing sentiment..."):
                start_time = time.time()
                
                # Get AI prediction
                result = analyzer(user_text)[0]
                
                # Calculate processing time
                processing_time = time.time() - start_time
            
            # Display results
            st.markdown("### 📊 Analysis Results")
            
            # Main metrics
            col_a, col_b, col_c = st.columns(3)
            
            # Determine emoji and color
            emoji = "😊" if result["label"] == "POSITIVE" else "😞"
            confidence_pct = result["score"] * 100
            
            with col_a:
                if result["label"] == "POSITIVE":
                    st.success(f"😊 **POSITIVE**")
                else:
                    st.error(f"😞 **NEGATIVE**")
            
            with col_b:
                st.metric("Confidence Score", f"{confidence_pct:.1f}%")
            
            with col_c:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            # Confidence visualization
            st.markdown("**Confidence Level:**")
            st.progress(result["score"])
            
            # Text stats
            col_x, col_y, col_z = st.columns(3)
            
            with col_x:
                st.metric("Text Length", f"{len(user_text)} characters")
            
            with col_y:
                word_count = len(user_text.split())
                st.metric("Word Count", f"{word_count} words")
            
            with col_z:
                confidence_level = "High" if result["score"] >= 0.8 else "Medium" if result["score"] >= 0.6 else "Low"
                st.metric("Confidence Level", confidence_level)
            
            # Detailed explanation
            if result["label"] == "POSITIVE":
                st.info(f"🎉 **Great news!** This text expresses positive sentiment with **{confidence_pct:.1f}%** confidence. The AI detected positive language patterns and emotional indicators.")
            else:
                st.warning(f"⚠️ **Heads up!** This text expresses negative sentiment with **{confidence_pct:.1f}%** confidence. The AI detected negative language patterns and emotional indicators.")
                
        else:
            st.warning("⚠️ Please enter some text to analyze!")

with col2:
    st.subheader("💡 Quick Examples")
    st.markdown("*Click to try these examples:*")
    
    examples = [
        ("😊 Positive", "This product is absolutely incredible! Best purchase I've ever made. Amazing quality and super fast delivery!"),
        ("😞 Negative", "Terrible experience! The product broke after one day and customer service was completely unhelpful. Total waste of money."),
        ("😐 Mixed", "The product works fine and delivery was on time, but nothing extraordinary. Just average quality for the price."),
        ("🤝 Service", "Outstanding customer support! They responded within minutes and solved my problem perfectly. Highly recommend!")
    ]
    
    for label, example in examples:
        if st.button(label, use_container_width=True):
            st.session_state.selected_example = example

# Handle example selection
if hasattr(st.session_state, 'selected_example'):
    st.rerun()

# Batch Analysis Section
st.markdown("---")
st.subheader("📊 Batch Analysis - Multiple Texts")

batch_input = st.text_area(
    "Enter multiple texts (one per line):",
    height=120,
    placeholder="""I love this amazing product!
Terrible quality, very disappointed.
The service was decent, nothing special.
Outstanding experience, highly recommend!
Poor value for money, not worth it."""
)

col_batch1, col_batch2 = st.columns(2)

with col_batch1:
    if st.button("📊 Analyze All Texts", type="secondary"):
        if batch_input.strip():
            # Parse texts
            texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            if texts:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                # Analyze each text
                for i, text in enumerate(texts):
                    status_text.text(f"Analyzing text {i+1} of {len(texts)}...")
                    
                    # Get prediction
                    result = analyzer(text)[0]
                    
                    # Store result
                    emoji = "😊" if result["label"] == "POSITIVE" else "😞"
                    results.append({
                        "Text": text[:60] + "..." if len(text) > 60 else text,
                        "Sentiment": f"{emoji} {result['label']}",
                        "Confidence": f"{result['score']:.1%}",
                        "Full_Text": text,
                        "Raw_Score": result['score']
                    })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(texts))
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show summary statistics
                st.markdown("### 📈 Summary Statistics")
                
                positive_count = sum(1 for r in results if "POSITIVE" in r["Sentiment"])
                negative_count = len(results) - positive_count
                avg_confidence = sum(r["Raw_Score"] for r in results) / len(results)
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                
                with col_s1:
                    st.metric("Total Texts", len(results))
                
                with col_s2:
                    st.metric("😊 Positive", f"{positive_count} ({positive_count/len(results)*100:.1f}%)")
                
                with col_s3:
                    st.metric("😞 Negative", f"{negative_count} ({negative_count/len(results)*100:.1f}%)")
                
                with col_s4:
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                # Results table
                st.markdown("### 📋 Detailed Results")
                
                display_df = pd.DataFrame([{
                    'Text Preview': r['Text'],
                    'Sentiment': r['Sentiment'],
                    'Confidence': r['Confidence']
                } for r in results])
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Download option
                full_df = pd.DataFrame([{
                    'Text': r['Full_Text'],
                    'Sentiment': r['Sentiment'].split()[-1],  # Remove emoji
                    'Confidence_Score': r['Raw_Score'],
                    'Confidence_Percentage': r['Confidence']
                } for r in results])
                
                csv = full_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("⚠️ No valid texts found. Please check your input.")
        else:
            st.warning("⚠️ Please enter some texts to analyze.")

with col_batch2:
    st.info("💡 **Batch Analysis Tips:**\n\n• Enter one text per line\n• Mix positive and negative examples\n• Check the summary statistics\n• Download results as CSV")

# Footer and sharing info
st.markdown("---")
st.markdown("""
### 🎉 Congratulations! Your AI App is LIVE!

**🌐 Share this app:** Your Replit automatically created a public URL - click the share button in Replit to get the link!

**📱 Mobile friendly:** This app works perfectly on phones and tablets too!

**🔄 Always online:** Your app is hosted in the cloud and accessible 24/7!
""")

# Sidebar information
st.sidebar.title("📊 App Information")

st.sidebar.markdown("""
### 🚀 Live on Replit!

**✨ Features:**
- Real-time AI sentiment analysis
- Single text processing
- Batch analysis capabilities  
- CSV data export
- Mobile-responsive design

**🤖 AI Model:**
- **Engine:** HuggingFace Transformers
- **Model:** DistilBERT
- **Accuracy:** ~91% on standard datasets
- **Speed:** <100ms per analysis

**📊 Project Status:**
- ✅ Day 1: Complete!
- 🔄 Day 2: Real-time streaming
- 📈 Day 3: Advanced charts
- 🎨 Day 4: Enhanced UI
- 🌐 Day 5: Full deployment
""")

st.sidebar.markdown("""
---
**🎯 Built in:** Python + Streamlit  
**🔥 Hosted on:** Replit  
**⏰ Started:** Aug 31, 2025, 10:29 PM IST  
**🏆 Status:** LIVE and WORKING! 🚀
""")
