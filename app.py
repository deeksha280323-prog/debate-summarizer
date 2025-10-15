import streamlit as st
from transformers import pipeline
import pandas as pd
from textblob import TextBlob

# ---- PAGE SETUP ----
st.set_page_config(page_title="Debate Summarizer", layout="wide")
st.title("🧠 NLP-Based Debate Summarizer with Sentiment Dashboard")

st.write("Upload or type in your debate transcript below to generate an AI summary and sentiment insights.")

# ---- INPUT ----
debate_text = st.text_area("🗣️ Paste debate text here:", height=200, placeholder="Type or paste the debate content...")

# ---- LOAD MODELS ----
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
            summary = summarizer(debate_text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

            # Sentiment Analysis using TextBlob
            blob = TextBlob(debate_text)
            sentiment_score = round(blob.sentiment.polarity, 2)
            if sentiment_score > 0:
                sentiment = "Positive"
            elif sentiment_score < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            # Display results
            st.subheader("📄 AI Summary:")
            st.success(summary)

            st.subheader("💬 Sentiment Overview:")
            st.metric(label="Sentiment Type", value=sentiment)
            st.metric(label="Sentiment Score", value=sentiment_score)

            # Consensus logic (very simple)
            consensus = "Strong Agreement" if sentiment_score > 0.4 else "Mixed Opinions" if sentiment_score > -0.2 else "Conflict Detected"
            st.subheader("🗳️ Consensus Analysis:")
            st.info(consensus)

            # Data Dashboard
            data = pd.DataFrame({
                "Category": ["Positive", "Negative", "Neutral"],
                "Score": [max(sentiment_score, 0), max(-sentiment_score, 0), 1 - abs(sentiment_score)]
            })
            st.bar_chart(data.set_index("Category"))

            # Export summary
            st.download_button("📥 Download Summary", summary, file_name="debate_summary.txt")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit, Hugging Face Transformers, and TextBlob")

