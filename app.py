import streamlit as st
from transformers import pipeline
import pandas as pd
from textblob import TextBlob
import plotly.express as px

# ---- PAGE SETUP ----
st.set_page_config(page_title="Debate Summarizer", layout="wide")
st.title("🧠 NLP-Based Debate Summarizer with Sentiment Dashboard")

st.write("Paste your debate transcript below to generate a summary, sentiment insights, and consensus analysis.")

# ---- INPUT ----
debate_text = st.text_area(
    "🗣️ Paste debate text here:",
    height=200,
    placeholder="Type or paste the debate content..."
)

# ---- LOAD SUMMARIZER ----
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ---- PROCESS ----
if st.button("🔍 Summarize Debate"):
    if debate_text.strip() == "":
        st.warning("Please enter some text before summarizing.")
    else:
        with st.spinner("Generating summary..."):
            # Summarization
            summary = summarizer(
                debate_text, max_length=130, min_length=30, do_sample=False
            )[0]["summary_text"]

            # Sentiment Analysis
            blob = TextBlob(debate_text)
            sentiment_score = round(blob.sentiment.polarity, 2)
            if sentiment_score > 0:
                sentiment = "Positive"
            elif sentiment_score < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            # Consensus logic
            consensus = (
                "Strong Agreement" if sentiment_score > 0.4
                else "Mixed Opinions" if sentiment_score > -0.2
                else "Conflict Detected"
            )

            # --- Dashboard Layout ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🗣️ Original Debate")
                with st.expander("View full debate"):
                    st.write(debate_text)

            with col2:
                st.subheader("📄 AI Summary")
                st.success(summary)

                st.subheader("💬 Sentiment Overview")
                st.metric("Type", sentiment)
                st.metric("Score", sentiment_score)

                st.subheader("🗳️ Consensus Analysis")
                st.info(consensus)

                # Pie chart for sentiment distribution
                data = pd.DataFrame({
                    "Category": ["Positive", "Negative", "Neutral"],
                    "Score": [max(sentiment_score, 0), max(-sentiment_score, 0), 1 - abs(sentiment_score)]
                })
                fig = px.pie(
                    data, names='Category', values='Score',
                    title='Sentiment Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download summary
                st.download_button("📥 Download Summary", summary, file_name="debate_summary.txt")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit, Hugging Face Transformers, and TextBlob")
