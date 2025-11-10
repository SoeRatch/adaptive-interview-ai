# Adaptive Interview AI  
**Dynamic Interview Question & Answer Generation using Topic Modeling + LLMs**

Adaptive Interview AI is an **AI-driven interview preparation system** that integrates **topic modeling (BERTopic)** and **Large Language Models (LLMs)** to dynamically generate technical interview questions, evaluate user responses, and provide adaptive feedback — simulating a personalized, AI-driven interview experience.

---

## Project Overview

This project demonstrates how modern **retrieval-augmented generation (RAG)** and **topic modeling** can be combined to build personalized interview preparation systems leveraging deep learning and natural language processing.

The system automatically builds a structured knowledge base from web content, discovers latent topics, and uses those topics to generate **context-aware interview questions**. User responses are evaluated by an **LLM**, which provides **real-time adaptive feedback** and **follow-up questions** until the answer quality improves - creating an engaging and interactive learning loop.

This project showcases expertise in:
- **Large Language Models (LLMs)** and **LangChain-based orchestration**  
- **Semantic Embeddings** and **Topic Modeling (BERTopic)**  
- **Vector Databases** (FAISS) for semantic retrieval  
- **PostgreSQL integration** for structured metadata  
- **End-to-End MLOps design** integrating data acquisition, preprocessing, modeling, and evaluation  


---

## Pipeline
1. **Web Discovery:** Identifies and scrapes high-quality technical content.
2. **Preprocessing:** Cleans, normalizes, and chunks text into coherent segments.
3. **Topic Modeling:** Generates embeddings and trains **BERTopic** to discover latent topics.
4. **Storage:** Saves semantic and structured data to **FAISS** (vector search) and **PostgreSQL** (metadata).
5. **Adaptive Q&A:** Generates LLM-driven interview questions and evaluates answers with real-time feedback.

---

## System Architecture

The system follows a **four-stage modular pipeline**:

### **1. Data Acquisition**
Collects and filters raw web text for domain-specific knowledge bases.  
- `url_discovery.py`: Discovers relevant URLs from a set of predefined seed links.
- `web_scrapper.py`: Extracts and filters clean text from those URLs.

### **2. Data Preprocessing**
Cleans and prepares text for embedding and topic modeling.
- `text_preprocessor.py`: Removes duplicates, non-English text, and formatting noise.
- `text_chunker.py`: Breaks large documents into semantically coherent chunks for embedding.

### **3. Topic Modeling**
Discovers latent topics and stores semantic representations.
- `embedding_generator.py`: Converts text chunks to dense embeddings using **SentenceTransformers**.
- `topic_model_trainer.py`: Trains **BERTopic** to cluster embeddings into interpretable topics.
- `topic_metadata_store.py`: Stores topic insights (IDs, keywords, document counts) into **PostgreSQL**.
- `topic_document_store.py`: Stores chunk-level embeddings into **FAISS** (for retrieval) and Postgres (for metadata).

### **4. Adaptive Question Answering**
Generates and evaluates interview questions dynamically.
- `question_generator.py`: Uses LLMs to generate interview questions, rubrics, and ideal answers.
- `answer_evaluator.py`: Evaluates candidate responses using LLM feedback loops with structured scoring.
- `interview_pipeline_cli.py`: CLI interface that orchestrates question generation, user interaction, and adaptive evaluation.

---

## Key Features

- **End-to-end modular pipeline:** From web data acquisition → NLP preprocessing → topic modeling → adaptive Q&A generation.

- **LLM-powered question generation & evaluation:** Dynamically crafted questions and rubric-based scoring.

- **BERTopic-driven topic discovery:** Groups semantically related content for focused question creation.

- **Hybrid storage system:**

    - PostgreSQL for structured topic metadata and document mapping.

    - FAISS / LangChain-compatible vector stores for semantic retrieval.

- **Adaptive learning feedback loop:** User answers are scored; follow-ups help improve responses in real time.

- **CLI-based interactive demo:** Experience a full AI-driven mock interview session.

---
## Workflow Summary

**Data Acquisition → Preprocessing → Topic Modeling → Adaptive Q&A**  
(URL Discovery → Web Scraping → Cleaning & Chunking → Embedding & BERTopic →  
Postgres + FAISS Storage → Question Generation + Answer Evaluation Loop)

The **interactive CLI** (`interview_pipeline_cli.py`) samples random topics, generates context-aware questions, and evaluates user responses iteratively — demonstrating a **fully automated, adaptive interview workflow**.


---

## Repository Structure
```
src/
├── data_acquisition/
│   └── web/
│       ├── url_discovery.py           # Discovers topic-relevant web pages
│       ├── web_scrapper.py            # Crawls and extracts textual content
│       ├── web_utils.py
│       ├── web_constants.py
│       └── content_filter.py
│
├── data_preprocessing/
│   ├── text_preprocessor.py           # Cleans and normalizes scraped text
│   ├── text_chunker.py                # Splits large docs into semantic chunks
│   └── preprocess_constants.py
│
├── topic_modeling/
│   ├── embedding_generator.py         # Creates embeddings using SentenceTransformer
│   ├── topic_model_trainer.py         # Trains BERTopic model
│   ├── topic_metadata_store.py        # Stores topic-level insights to Postgres
│   ├── topic_document_store.py        # Stores doc-topic mappings in Postgres & FAISS
│   ├── embedding_storage.py
│   └── constants.py
│
├── rag/
│   ├── question_generator.py          # Generates LLM-based interview questions
│   ├── answer_evaluator.py            # Evaluates user answers and gives feedback
│   ├── interview_pipeline_cli.py      # Orchestrates full adaptive interview flow
│   └── utils.py
│
└── database/
    ├── postgres_handler.py
    ├── setup_schema.py
    ├── vector_db/
    │   ├── faiss_handler.py
    │   ├── langchain_base.py
    │   ├── qdrant_handler.py
    │   └── pinecone_handler.py
    └── vector_db_factory.py

data/
├── raw/ → processed/ → embeddings/ → models/ → vector_index/

```
---
## Why This Project Stands Out

- Demonstrates **full-stack AI engineering** — from data collection to adaptive LLM evaluation.  
- Bridges **classical NLP (topic modeling)** with **modern LLM-based generation**.  
- Showcases **scalable, production-ready design** with modular components.  
- Applies real-world **RAG principles** and **semantic retrieval pipelines**. 

---

## Author

[SoeRatch](https://github.com/SoeRatch)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
