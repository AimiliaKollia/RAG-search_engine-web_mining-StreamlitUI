# RAG-search_engine-web_mining-StreamlitUI
Web mining, information retrieval (BM25+, TF-IDF, embeddings, semantic search with reranking) and RAG pipeline with Streamlit UI using Wikipedia data.  

## Team
- AimiliaKollia
- NancyLabropoulou
- KyriakiSkiada

- ## Overview
This project implements a full **Search Engine and Web Mining pipeline** that combines classical information retrieval techniques with modern semantic search and **Retrieval-Augmented Generation (RAG)**.  
The system is designed for **academic and educational purposes**, focusing on structured retrieval and analysis of information related to serial killers using Wikipedia as the data source.

The pipeline integrates:
- Web scraping and corpus construction
- Lexical retrieval (BM25+, TF-IDF)
- Dense semantic retrieval (Sentence Transformers)
- Hybrid filtering with a vector database (Qdrant)
- Cross-encoder re-ranking
- Large Language Model (LLM) generation
- An interactive Streamlit-based conversational interface

---

## Key Features
- 🔍 **Hybrid Retrieval**: Combines lexical (BM25+, TF-IDF) and semantic (MPNet embeddings) search
- 🧠 **Semantic Search** using transformer-based embeddings
- ⚡ **Cross-Encoder Re-ranking** for improved precision
- 🧾 **Structured Metadata Filtering** (country, victim counts, years active)
- 💬 **Retrieval-Augmented Generation (RAG)** with controlled prompting
- 🌐 **Interactive Streamlit Application**
- 📊 **Transparent Debug Panels** showing retrieval and filtering steps


DISCLAIMER! You will need a valid Hugging Face Access Token to run the two files and you will also need access to a Qdrant database.
---

## Data Collection
- Source: **Wikipedia – “List of serial killers by number of victims”**
- Extracted data from 7 tables (top 40 entries per table)
- Additional biographies scraped from individual Wikipedia pages
- Total entries: **274 serial killers**
- Total text chunks: **2,254**

### Collected Fields
- Name
- Country
- Years active
- Proven victims
- Possible victims
- Notes and full biography text

### Ethical Scraping Measures
- Rotating user agents
- 1-second request delay
- Content filtering (minimum 50 characters)

---

## Preprocessing
Two separate pipelines were implemented:

### Semantic (RAG / Embeddings)
- Minimal preprocessing
- Preserved punctuation, casing, and sentence structure
- Whitespace normalization only

### Lexical (BM25+ / TF-IDF)
- Lowercasing
- Stopword removal (with semantic exceptions)
- Lemmatization using spaCy (`en_core_web_sm`)
- Tokenization
- Short-word filtering
- Numeric normalization for victim counts

---

## Retrieval Models
### Lexical
- **BM25+**
- **TF-IDF** (unigrams & bigrams, cosine similarity)

### Semantic
- **Sentence Transformers**: `all-mpnet-base-v2`
- Embedding dimension: 768

---

## Vector Database
- **Qdrant Cloud**
- Cosine similarity
- Indexed metadata fields:
  - `name`
  - `country`
  - `proven_victims`
  - `possible_victims`

Supports hybrid queries such as:
> “Serial killers from Russia with more than 20 proven victims”

---

## Cross-Encoder Re-ranking
- Model: `ms-marco-MiniLM-L-6-v2`
- Re-ranks top retrieved results based on query–document pairs
- Improves contextual relevance before RAG generation

---

## Retrieval-Augmented Generation (RAG)
- LLM: **Qwen2.5-72B-Instruct** (Hugging Face API)
- Controlled prompting:
  - Answers strictly derived from retrieved context
  - No external knowledge
  - Concise responses (≤150 words)
- Structured context blocks per entity

---

## Streamlit Application
The final system is deployed as an **interactive Streamlit web app** featuring:
- Conversational interface
- Hybrid retrieval & filtering
- Debug panels showing:
  - Retrieved context
  - Applied filters
  - Re-ranking scores
- Educational disclaimer and transparency

---

## Educational Disclaimer
**Educational Prototype**

Developed for the course **Search Engine and Web Mining**  
**MSc in Data Science**  
**American College of Greece | Deree**

- 👩‍💻 Team: Aimilia, Nancy, Kyriaki  
- 👨‍🏫 Instructor: Dr. Lazaros Polymenakos  

⚠️ This project is intended **for academic demonstration only**.  
It is not designed for real-world, commercial, or investigative use.

---

## 📁 Project Files

- `Search_Engine_code.ipynb` — Notebook with the web mining and preprocessing pipeline.
- `app.py` — Streamlit application for RAG chatbot.


## Technologies Used
- Python
- spaCy
- SentenceTransformers
- Qdrant
- Hugging Face API
- Streamlit
- Scikit-learn

---

## Data Source
Wikipedia – *List of serial killers by number of victims*
https://en.wikipedia.org/wiki/List_of_serial_killers_by_number_of_victims
---

## License
This project is provided for **educational and research purposes only**.
