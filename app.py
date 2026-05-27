import streamlit as st
import os
import re
import time
from summarizer import ExtractiveSummarizer, AbstractiveSummarizer

# 1. Page Configuration and Layout
st.set_page_config(
    page_title="AI News Article Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if 'article_input' not in st.session_state:
    st.session_state.article_input = ""
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# 2. Premium Design CSS Injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;700;800&display=swap');

/* Global Font styling */
.main, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App Title styling */
.app-title-container {
    padding: 1.5rem 0 0.5rem 0;
    text-align: left;
}

.app-title {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #10B981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    margin-bottom: 0.1rem;
    letter-spacing: -0.06rem;
}

.app-subtitle {
    font-size: 1.05rem;
    color: #64748B;
    margin-bottom: 1.5rem;
    font-weight: 400;
}

/* Glassmorphic Cards for Analytics Dashboard */
.metrics-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: rgba(128, 128, 128, 0.05);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 12px;
    padding: 1.1rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s, border-color 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(108, 99, 255, 0.4);
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.08);
}

.metric-card.highlight {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
    border: 1px solid rgba(108, 99, 255, 0.25);
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06rem;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-color, #1F2937);
    font-family: 'Outfit', sans-serif;
    line-height: 1.2;
}

.metric-unit {
    font-size: 0.8rem;
    font-weight: 500;
    color: #64748B;
}

.metric-sub {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 0.2rem;
}

/* Beautiful custom section header */
.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-color, #1F2937);
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
    border-bottom: 2px solid rgba(108, 99, 255, 0.2);
    padding-bottom: 0.3rem;
    display: inline-block;
}

/* Premium summary output block */
.summary-output {
    background-color: var(--background-color, rgba(128, 128, 128, 0.02));
    border-left: 4px solid #6C63FF;
    border-radius: 4px;
    padding: 1.25rem;
    font-size: 1.05rem;
    line-height: 1.7;
    color: var(--text-color, #1F2937);
    margin: 1rem 0;
}

/* Visual Progress Bar */
.progress-container {
    background-color: rgba(128, 128, 128, 0.1);
    border-radius: 10px;
    height: 10px;
    width: 100%;
    margin: 0.5rem 0 1rem 0;
    overflow: hidden;
}

.progress-bar-fill {
    background: linear-gradient(90deg, #6C63FF 0%, #3B82F6 100%);
    height: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# 3. Model Loading & Caching resource
@st.cache_resource
def get_extractive_summarizer():
    return ExtractiveSummarizer()

@st.cache_resource
def get_abstractive_summarizer(model_name):
    return AbstractiveSummarizer(model_name=model_name)

# 4. Helper callbacks for Session State manipulation
def load_sample_callback():
    try:
        with open("sample_article.txt", "r", encoding="utf-8") as f:
            st.session_state.article_input = f.read()
            st.session_state.summary_result = None  # Reset previous summaries
    except Exception as e:
        st.error(f"Error loading sample file: {e}")

def clear_input_callback():
    st.session_state.article_input = ""
    st.session_state.summary_result = None

# --- SIDEBAR: Configuration Controls ---
st.sidebar.markdown("### ⚙️ Summarizer Settings")

# Algorithm Choice
method = st.sidebar.radio(
    "Select Summarization Type",
    options=["Extractive (TextRank)", "Abstractive (Transformer)"],
    help="Extractive selects key original sentences. Abstractive drafts a paraphrased summary using an AI model."
)

if method == "Extractive (TextRank)":
    num_sentences = st.sidebar.slider(
        "Summary Length (Sentences)",
        min_value=2,
        max_value=15,
        value=5,
        step=1,
        help="Select the exact number of key sentences to extract from the article."
    )
else:
    target_words = st.sidebar.slider(
        "Target Length (Words)",
        min_value=50,
        max_value=500,
        value=150,
        step=10,
        help="Choose the approximate word length of the generated summary."
    )
    
    model_choice = st.sidebar.selectbox(
        "Transformer Model",
        options=["t5-small", "facebook/bart-base"],
        index=0,
        help="t5-small (~242MB) is lightweight and fast on CPU. facebook/bart-base (~558MB) produces slightly higher-quality output but takes longer to load."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Actions")
st.sidebar.button("Load Sample Article", on_click=load_sample_callback, use_container_width=True)
st.sidebar.button("Clear Input Text", on_click=clear_input_callback, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.8rem; color:#64748B; text-align:center;'>Powered by Custom TextRank & Hugging Face Transformers</div>", 
    unsafe_allow_html=True
)

# --- MAIN PANEL ---

# App Header
st.markdown("""
<div class="app-title-container">
    <div class="app-title">AI News Summarizer</div>
    <div class="app-subtitle">Generate clear, coherent summaries of long news articles using extractive or abstractive artificial intelligence.</div>
</div>
""", unsafe_allow_html=True)

# Main Text Input Area
# Tie the value to st.session_state.article_input so buttons can modify it
article_text = st.text_area(
    "Paste your news article here:",
    value=st.session_state.article_input,
    height=280,
    placeholder="Paste a long news article, editorial, or report (3,000+ words supported)...",
    key="article_text_area_input"
)
st.session_state.article_input = article_text # update state

# Trigger Button
col_btn, col_info = st.columns([1, 4])
with col_btn:
    run_summarization = st.button("✨ Summarize Article", type="primary", use_container_width=True)
with col_info:
    # Quick indicator of input length
    word_count = len(article_text.split())
    if word_count > 0:
        st.markdown(f"<div style='margin-top: 8px; color: #64748B;'>Detected <b>{word_count}</b> words. Ready for summarization.</div>", unsafe_allow_html=True)

# --- PROCESSING ENGINE ---
if run_summarization:
    if not article_text.strip():
        st.warning("⚠️ The input text area is empty. Please paste an article first or load the sample article.")
    else:
        # Pre-process text verification
        cleaned_text = article_text.strip()
        
        # Extractive Method
        if method == "Extractive (TextRank)":
            with st.spinner("Analyzing text connections and extracting key sentences via TextRank..."):
                try:
                    t_start = time.time()
                    summarizer = get_extractive_summarizer()
                    summary = summarizer.summarize(cleaned_text, num_sentences=num_sentences)
                    t_end = time.time()
                    
                    st.session_state.summary_result = {
                        "text": summary,
                        "type": "Extractive (TextRank)",
                        "model": "Custom Python Implementation",
                        "time_taken": t_end - t_start,
                        "sentences_count": num_sentences
                    }
                except Exception as e:
                    st.error(f"Error during extractive summarization: {e}")
                    
        # Abstractive Method
        else:
            # Let the user know the model is loading
            loading_msg = f"Initializing Hugging Face model '{model_choice}'..."
            if "t5" in model_choice:
                loading_msg += " (First run will download ~242MB)"
            else:
                loading_msg += " (First run will download ~558MB)"
                
            with st.spinner(loading_msg):
                try:
                    t_start = time.time()
                    summarizer = get_abstractive_summarizer(model_choice)
                    
                    # Ensure the model compiles/loads successfully
                    success, msg = summarizer.load_model()
                    if not success:
                        st.error(f"Model Loading Error: {msg}")
                    else:
                        summary = summarizer.summarize(cleaned_text, target_words=target_words)
                        t_end = time.time()
                        
                        st.session_state.summary_result = {
                            "text": summary,
                            "type": "Abstractive (Transformer)",
                            "model": model_choice,
                            "time_taken": t_end - t_start,
                            "target_words": target_words
                        }
                except Exception as e:
                    st.error(f"Error during abstractive summarization. Please check model download or memory capability. Details: {e}")

# --- RESULTS DASHBOARD ---
if st.session_state.summary_result:
    summary_data = st.session_state.summary_result
    summary_text = summary_data["text"]
    
    # Calculate Metrics
    orig_words = len(article_text.split())
    orig_chars = len(article_text)
    
    summ_words = len(summary_text.split())
    summ_chars = len(summary_text)
    
    # Handle zero division safety
    compression_ratio = ((orig_words - summ_words) / orig_words * 100) if orig_words > 0 else 0
    compression_ratio = max(0.0, compression_ratio) # clamp to 0
    
    # Est Reading Time: WPM = 200
    orig_reading_time = orig_words / 200
    summ_reading_time = summ_words / 200
    time_saved = max(0.0, orig_reading_time - summ_reading_time)
    
    st.markdown("<div class='section-header'>📊 Summarization Dashboard</div>", unsafe_allow_html=True)
    
    # Premium Metrics Widgets
    st.markdown(f"""
    <div class="metrics-container">
        <div class="metric-card">
            <div class="metric-label">Original Length</div>
            <div class="metric-value">{orig_words} <span class="metric-unit">words</span></div>
            <div class="metric-sub">{orig_chars:,} characters</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Summary Length</div>
            <div class="metric-value">{summ_words} <span class="metric-unit">words</span></div>
            <div class="metric-sub">{summ_chars:,} characters</div>
        </div>
        <div class="metric-card highlight">
            <div class="metric-label">Compression Ratio</div>
            <div class="metric-value">{compression_ratio:.1f}%</div>
            <div class="metric-sub">Shorter than original</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Time Saved</div>
            <div class="metric-value">{time_saved:.1f} <span class="metric-unit">min</span></div>
            <div class="metric-sub">At average 200 WPM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual comparison bar
    percent_summary_size = 100 - compression_ratio
    st.markdown("##### 📏 Size Comparison (Summary vs Original Document)")
    st.markdown(f"""
    <div class="progress-container" title="Summary represents {percent_summary_size:.1f}% of the original document">
        <div class="progress-bar-fill" style="width: {percent_summary_size:.1f}%;"></div>
    </div>
    <div style="font-size: 0.8rem; color: #64748B; margin-top: -8px; margin-bottom: 1.5rem;">
        Summary represents <b>{percent_summary_size:.1f}%</b> of the original article.
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for Summary View vs Metadata Analysis
    tab1, tab2 = st.tabs(["📄 Generated Summary", "🔍 Details & Key Insights"])
    
    with tab1:
        st.markdown(f"""
        <div class="summary-output">
            {summary_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Easy Copy Tool (Uses standard code block widget for built-in copying)
        st.caption("📋 Copy summary text:")
        st.code(summary_text, language="text")
        
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📌 Summarization Metadata")
            st.write(f"**Method Selected:** {summary_data['type']}")
            st.write(f"**Engine/Model:** `{summary_data['model']}`")
            st.write(f"**Execution Speed:** {summary_data['time_taken']:.3f} seconds")
            if "sentences_count" in summary_data:
                st.write(f"**Sentences Extracted:** {summary_data['sentences_count']}")
            if "target_words" in summary_data:
                st.write(f"**Requested Target Word Count:** ~{summary_data['target_words']}")
        with col2:
            st.markdown("##### 🔑 Key Points")
            # Extract list of points by breaking down summary text
            try:
                sentences = nltk.sent_tokenize(summary_text)
            except Exception:
                sentences = re.split(r'(?<=[.!?])\s+', summary_text)
            
            # Show up to 5 points
            for i, sent in enumerate(sentences[:5]):
                if sent.strip():
                    st.markdown(f"**{i+1}.** {sent.strip()}")
            if len(sentences) > 5:
                st.markdown(f"*And {len(sentences) - 5} more sentence(s)...*")
                
        # Debugging / Chunking overview for transparency
        with st.expander("🔬 Technical Diagnostics: Document Chunking Analysis"):
            chunks = chunk_text(article_text, max_chunk_words=350)
            st.write(f"**Total chunks parsed:** {len(chunks)}")
            for idx, chunk in enumerate(chunks):
                word_len = len(chunk.split())
                st.write(f"**Chunk {idx + 1}:** {word_len} words | *Snippet:* `{chunk[:80]}...`")
                
else:
    # Welcome card if no summary has been generated yet
    st.info("💡 Paste a news article above or load our pre-configured sample article, then click **Summarize Article** to run the summarization models.")
