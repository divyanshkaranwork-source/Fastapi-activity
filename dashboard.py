"""Assessment-1 dashboard.  Run:  streamlit run dashboard.py"""

import json

import pandas as pd
import streamlit as st

from app.analyzer import InvalidModelOutput, analyze
from app.llm import LLMError

st.set_page_config(page_title="Assessment-1", page_icon="🏦")
st.title("Assessment-1: bank feedback analyzer")
st.caption("Paste a customer message. The LLM returns validated JSON: sentiment, urgency, action, confidence.")

SAMPLES = {
    "(type your own)": "",
    "Fraud": "I got an SMS for a Rs 48,000 debit at a shop I never visited. Block my card now!",
    "Double charge": "My card was charged twice for Rs 3,200 at a petrol pump. Please reverse it.",
    "Praise": "The branch staff were wonderful and my KYC was done in ten minutes. Thank you!",
}
if "history" not in st.session_state:
    st.session_state.history = []

sample = st.selectbox("Sample message", list(SAMPLES))
text = st.text_area("Customer feedback", value=SAMPLES[sample], height=140, key=f"text_{sample}")

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.error("Please enter some feedback.")
    else:
        try:
            with st.spinner("Analyzing..."):
                result = analyze(text)
            st.session_state.history.append({"feedback": text[:80], **result.model_dump()})
            c1, c2, c3 = st.columns(3)
            c1.metric("Sentiment", result.sentiment)
            c2.metric("Urgency", result.urgency)
            c3.metric("Confidence", f"{result.confidence:.0%}")
            st.write("**Recommended action:**", result.recommended_action)
            st.code(json.dumps(result.model_dump(), indent=2), language="json")
        except ValueError as exc:
            st.error(str(exc))
        except InvalidModelOutput as exc:
            st.error(f"The model returned invalid output: {exc}")
        except LLMError as exc:
            st.error(f"LLM error: {exc}")

history = st.session_state.history
if history:
    st.divider()
    st.subheader("Dashboard")
    df = pd.DataFrame(history)
    left, right = st.columns(2)
    left.write("Sentiment")
    left.bar_chart(df["sentiment"].value_counts())
    right.write("Urgency")
    right.bar_chart(df["urgency"].value_counts())
    st.write(f"Average confidence: **{df['confidence'].mean():.0%}** over {len(df)} analyses")
    st.dataframe(df, hide_index=True)
