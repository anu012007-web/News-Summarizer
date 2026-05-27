import re
import math
import numpy as np
import nltk

def download_nltk_resources():
    """Download required NLTK datasets if they are not already available."""
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for res in resources:
        try:
            if res == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif res == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif res == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download NLTK resource '{res}': {e}")

# Run NLTK download on module import
download_nltk_resources()

def chunk_text(text, max_chunk_words=350):
    """
    Split the input text into chunks of sentences such that each chunk
    does not exceed max_chunk_words. This prevents out-of-memory and context
    limit errors for abstractive summarization.
    """
    if not text.strip():
        return []
        
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback if sentence tokenizer fails
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        num_words = len(words)
        
        # If a single sentence exceeds the limit, place it in its own chunk
        if num_words > max_chunk_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(sentence)
        elif current_word_count + num_words > max_chunk_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = num_words
        else:
            current_chunk.append(sentence)
            current_word_count += num_words
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks


class ExtractiveSummarizer:
    """
    Extractive summarization using a custom scratch-built TextRank algorithm.
    """
    def __init__(self):
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
            
    def _clean_sentence_words(self, sentence):
        """Helper to tokenize, lowercase, and clean a sentence into a set of content words."""
        try:
            words = nltk.word_tokenize(sentence.lower())
        except Exception:
            # Fallback if word tokenizer fails
            words = re.findall(r'\b\w+\b', sentence.lower())
            
        # Filter stopwords and non-alphanumeric tokens
        content_words = {w for w in words if w.isalnum() and w not in self.stop_words}
        return content_words

    def _sentence_similarity(self, sent1_words, sent2_words):
        """
        Calculate TextRank similarity between two sentences based on word overlap.
        Similarity(S1, S2) = |S1 ∩ S2| / (log(|S1|) + log(|S2|))
        """
        if not sent1_words or not sent2_words:
            return 0.0
            
        overlap = len(sent1_words & sent2_words)
        if overlap == 0:
            return 0.0
            
        len1 = len(sent1_words)
        len2 = len(sent2_words)
        
        # Avoid log(1) = 0 division by zero
        if len1 <= 1 or len2 <= 1:
            return 0.0
            
        denom = math.log(len1) + math.log(len2)
        return overlap / denom

    def _compute_pagerank(self, similarity_matrix, d=0.85, max_iter=100, tol=1e-6):
        """
        Compute the PageRank scores of sentences using power iteration.
        similarity_matrix: 2D numpy array representing sentence similarities.
        """
        n = similarity_matrix.shape[0]
        if n == 0:
            return np.array([])
            
        # Sum of outgoing edges (similarity values) for each sentence
        row_sums = np.sum(similarity_matrix, axis=1)
        
        # Build the transition matrix M where M[j, i] is transition from i to j
        M = np.zeros((n, n))
        for i in range(n):
            if row_sums[i] > 0:
                M[:, i] = similarity_matrix[i, :] / row_sums[i]
            else:
                # Dangling node: distribute probability uniformly
                M[:, i] = np.ones(n) / n
                
        # Initialize uniform probability distribution vector
        v = np.ones(n) / n
        
        for _ in range(max_iter):
            v_next = d * np.dot(M, v) + (1 - d) / n * np.ones(n)
            # Check for convergence
            if np.linalg.norm(v_next - v, 1) < tol:
                v = v_next
                break
            v = v_next
            
        return v

    def summarize(self, text, num_sentences=5):
        """
        Summarize text extractively using TextRank.
        text: Input document string.
        num_sentences: Number of sentences to select for the summary.
        """
        if not text.strip():
            return ""
            
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        n = len(sentences)
        
        if n <= num_sentences:
            return " ".join(sentences)
            
        # Precompute words set for each sentence
        sentence_words = [self._clean_sentence_words(s) for s in sentences]
        
        # Build Similarity Matrix
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(sentence_words[i], sentence_words[j])
                    
        # Compute PageRank scores
        scores = self._compute_pagerank(similarity_matrix)
        if len(scores) == 0:
            return " ".join(sentences[:num_sentences])
            
        # Sort sentences based on scores
        ranked_sentences = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        # Select top sentences
        selected_indices = [idx for idx, score in ranked_sentences[:num_sentences]]
        
        # Re-sort indices to output sentences in their original order
        selected_indices.sort()
        
        summary = [sentences[idx] for idx in selected_indices]
        return " ".join(summary)


class AbstractiveSummarizer:
    """
    Abstractive summarization using Hugging Face sequence-to-sequence models (T5, BART).
    Loads tokenizer and model directly to maintain compatibility across all Hugging Face versions.
    """
    def __init__(self, model_name="t5-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        
    def load_model(self):
        """Lazy load Hugging Face model and tokenizer with error handling."""
        if self.model is not None and self.tokenizer is not None:
            return True, "Model already loaded."
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Select GPU if available, else CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load tokenizer and model weights
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            return True, "Model loaded successfully."
        except Exception as e:
            return False, f"Error initializing model '{self.model_name}': {str(e)}"

    def summarize(self, text, target_words=150):
        """
        Summarize text abstractively using chunking and direct PyTorch model generation.
        text: Input document string.
        target_words: Approximate length of the reconstructed summary in words.
        """
        if not text.strip():
            return ""
            
        # Lazy load model
        success, message = self.load_model()
        if not success:
            raise RuntimeError(message)
            
        # Chunk text into blocks of max 350 words (leaves room for model's token limits)
        chunks = chunk_text(text, max_chunk_words=350)
        num_chunks = len(chunks)
        
        if num_chunks == 0:
            return ""
            
        chunk_summaries = []
        for chunk in chunks:
            chunk_word_count = len(chunk.split())
            
            # If the chunk is very short, keep it as is (summarizing short blocks is unstable)
            if chunk_word_count < 35:
                chunk_summaries.append(chunk)
                continue
                
            # Distribute the target word count across all chunks
            target_words_for_chunk = max(20, target_words // num_chunks)
            if target_words_for_chunk >= chunk_word_count:
                chunk_summaries.append(chunk)
                continue
                
            # Convert words to tokens estimate (1 word ≈ 1.3 tokens)
            max_tokens = int(target_words_for_chunk * 1.3)
            min_tokens = int(target_words_for_chunk * 0.7)
            
            # Safe boundaries to avoid pipeline crashes
            min_tokens = max(5, min_tokens)
            max_tokens = max(min_tokens + 10, max_tokens)
            
            # Ensure max_tokens is less than input length
            input_tokens = int(chunk_word_count * 1.3)
            if max_tokens >= input_tokens:
                max_tokens = max(input_tokens - 5, min_tokens + 5)
            if min_tokens >= max_tokens:
                min_tokens = max(5, max_tokens - 5)
                
            # Add summarize task prefix if using a T5 model
            prefix = "summarize: " if "t5" in self.model_name.lower() else ""
            input_text = prefix + chunk
            
            try:
                # Tokenize inputs
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                # Move to correct device (GPU/CPU)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Perform model generation
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_tokens,
                    min_length=min_tokens,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode and cleanup
                chunk_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                chunk_summaries.append(chunk_summary)
            except Exception as e:
                # Fallback to original chunk if processing fails
                print(f"Warning: Chunk summarization failed. Falling back to original chunk. Details: {e}")
                chunk_summaries.append(chunk)
                
        # Reconstruct cohesive text from chunk summaries
        combined_summary = " ".join(chunk_summaries)
        
        # Clean up whitespace/double-spaces
        combined_summary = re.sub(r'\s+', ' ', combined_summary).strip()
        return combined_summary
