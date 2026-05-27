# 📰 AI News Article Summarizer

A complete, premium web application built in Python using **Streamlit** to generate coherent and concise summaries of long-form news articles. The application supports both **extractive** summarization (using a custom TextRank algorithm implemented from scratch) and **abstractive** summarization (using Hugging Face seq-to-seq transformer models). It handles articles of any length (including 3,000+ word editorials) through an intelligent sentence-level text chunking pipeline.

---

## ✨ Features

- **Extractive Summarization**: A custom implementation of the **TextRank** (PageRank-based) algorithm built from scratch. It constructs a sentence similarity network based on content word overlap and computes stationary distribution scores via power iteration to extract the most key original sentences.
- **Abstractive Summarization**: Powered by Hugging Face sequence-to-sequence transformers like `t5-small` or `facebook/bart-base` to draft paraphrased, human-like summaries.
- **Intelligent Chunking Engine**: Automatically splits long articles (e.g. 3,000+ words) into sentence-aligned blocks of up to 350 words, summarizes each block proportionally, and reconstructs a cohesive summary, avoiding context limit overflows and memory crashes.
- **Premium Analytics Dashboard**:
  - **Live Metrics**: Measures original vs. summary word/character counts, compression percentages, and reading time saved (based on 200 WPM average).
  - **Size Comparison Bar**: A graphical visualizer showing the summary size relative to the original document.
  - **Key Insights Tab**: Extracts the main bullet points from the generated summary.
  - **Technical Diagnostics**: Details the text chunks parsed and word distributions.
- **Modern User Interface**: Styled with customized Outfit & Inter typography, glassmorphism card layouts, and responsive components.

---

## 📂 Project Structure

```
├── app.py              # Streamlit dashboard and UI code
├── summarizer.py       # Core classes: ExtractiveSummarizer, AbstractiveSummarizer
├── requirements.txt    # Project dependencies
└── sample_article.txt  # 1,000-word sample news article for testing
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/anu012007-web/News-Summarizer.git
   cd News-Summarizer
   ```

2. **Install Dependencies**:
   It is recommended to run the app in a virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: NLTK datasets (`punkt`, `stopwords`, and `punkt_tab`) are downloaded automatically on the first run.*

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to the address shown in your terminal (typically **`http://localhost:8501`**).

---

## 🧠 How It Works

### Extractive Summarizer (TextRank)
1. The text is split into sentences $S$.
2. Every sentence is tokenized and cleaned by removing stopwords and non-alphanumeric words to form a set of content words.
3. A similarity matrix is built where the weight between two sentences $S_i$ and $S_j$ is computed as:
   $$Similarity(S_i, S_j) = \frac{|\{w \in S_i \cap S_j\}|}{\log(|S_i|) + \log(|S_2|)}$$
4. The PageRank scores of sentences are computed using the power iteration method:
   $$v_{next} = d \cdot M \cdot v + \frac{1-d}{N} \mathbf{1}$$
   where $d = 0.85$ (damping factor) and $M$ is the column-normalized transition matrix.
5. The sentences are sorted by their rank, and the top $K$ sentences are returned in their original order of appearance to maintain reading coherence.

### Abstractive Summarizer (Transformers)
- Because newer Hugging Face versions have modified the `pipeline("summarization")` schema, this model loads classes directly using `AutoTokenizer` and `AutoModelForSeq2SeqLM` to ensure compatibility across all environments.
- Large texts are chunked at sentence boundaries to fit within model context windows (e.g. 512 tokens).
- Dynamic token generation length is calculated proportionally:
  $$\text{target\_words\_for\_chunk} = \max\left(20, \frac{\text{total\_target\_words}}{\text{num\_chunks}}\right)$$
  This budget is converted to target generation tokens (`max_length` / `min_length`) with defensive safety guards to prevent model execution crashes on short inputs.
