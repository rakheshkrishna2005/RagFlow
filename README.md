# 🚀 RagFlow

RagFlow is a Retrieval Augmented Generation (RAG) API that processes PDF documents and answers questions using advanced language models.

## 📑 Table of Contents
- [✨ Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [🔄 RAG Pipeline Architecture](#-rag-pipeline-architecture)
- [🔑 Environment Variables](#-environment-variables)
- [🔐 Authentication](#-authentication)
- [📡 API Endpoints](#-api-endpoints)
- [❌ Error Responses](#-error-responses)
- [⚡ Rate Limiting & Performance](#-rate-limiting--performance)

## 🎯 Features

- Hybrid search combining dense and sparse embeddings
- Support for both local PDF files and remote URLs
- Automatic text chunking with overlap for context preservation
- Parallel processing of document embeddings
- Retry logic with exponential backoff for API calls
- Comprehensive error handling and logging

## 🛠️ Tech Stack

### 🔧 Backend Framework and API
- FastAPI - High-performance async web framework
- Uvicorn - ASGI server implementation
- CORS middleware for cross-origin support

### 🧠 Machine Learning and Embeddings
- Google's Gemini model (models/embedding-001) for document embeddings
- Groq LLM (llama-3.3-70b-versatile) for answer generation
- FAISS for efficient similarity search using cosine similarity
- TF-IDF vectorizer for sparse lexical matching
- NumPy for numerical operations and vector manipulations

### 📄 Document Processing
- PyPDF2 for PDF text extraction
- Custom text chunking with configurable overlap
- Parallel processing using ThreadPoolExecutor and asyncio

### 🛠️ Utilities and Error Handling
- Python-dotenv for environment management
- Tenacity for retry logic and backoff
- Pydantic for request/response validation
- Logging for production monitoring

## 🔄 RAG Pipeline Architecture

### 1. Document Ingestion
- Supports both local PDF files and remote URLs
- Extracts text content from PDF documents
- Splits text into overlapping chunks (750 chars with 150 char overlap)

### 2. Embedding Generation
- Processes text chunks in parallel using ThreadPoolExecutor
- Generates dense embeddings using Google's Gemini model
- Creates sparse embeddings using TF-IDF vectorization
- Normalizes vectors using L2 normalization for cosine similarity

### 3. Hybrid Search
- Combines dense and sparse search methods:
  - Dense: Semantic search using FAISS (Inner Product similarity)
  - Sparse: Lexical matching using TF-IDF
- Weighted combination (default: 0.7 dense, 0.3 sparse)
- Returns top K most relevant chunks (default: K=5)

### 4. Answer Generation
- Takes user question and retrieved context chunks
- Limited to 5 most relevant chunks for context window management
- Uses Groq's LLM for answer generation
- Implements retry logic with exponential backoff
- Returns concise, plain text answers

### 5. Performance Optimizations
- Parallel processing of document embeddings
- Efficient vector similarity search with FAISS
- Automatic chunking for optimal context windows
- Request/response validation using Pydantic
- Comprehensive error handling and logging

### ⚙️ System Configuration
```python
# Embedding Configuration
EMBED_DIM = 768  # Dimensionality of Gemini embeddings

# Text Chunking Parameters
CHUNK_SIZE = 750  # Characters per chunk
CHUNK_OVERLAP = 150  # Overlap between chunks

# Processing Parameters
BATCH_SIZE = 10  # Batch size for embedding generation
TOP_K = 5  # Number of relevant chunks to retrieve
MAX_DOCS = 5  # Maximum chunks in LLM context
```

## 🔑 Environment Variables

### API Server (.env)
```env
API_KEY=your_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### Frontend (.env)
```env
API_KEY=your_api_key_here
```

## 🔐 Authentication

All API endpoints require Bearer token authentication.

```http
Authorization: Bearer your_api_key_here
```

## 📡 API Endpoints

### 📥 Get API Information
```http
GET /
```

#### 📤 Response
```json
{
    "name": "RAG API Server",
    "version": "1.0.0",
    "description": "A REST API for Question Answering using RAG (Retrieval Augmented Generation)",
    "endpoints": {
        "/": {
            "method": "GET",
            "description": "Get API information"
        },
        "/run": {
            "method": "POST",
            "description": "Process documents and answer questions using RAG",
            "authentication": "Bearer token required"
        }
    },
    "components": {
        "embedding_model": "models/embedding-001",
        "llm_model": "llama-3.3-70b-versatile"
    }
}
```

### Logs
```
🚀 » New request received from client
📑 » Loading PDF from web source: https://arxiv.org/pdf/2005.11401.pdf
✨ » PDF loaded successfully
📑 » Loading PDF from web source: https://arxiv.org/pdf/1706.03762.pdf
✨ » PDF loaded successfully
📝 » Extracting text content
✂️  » Text split into 137 smart chunks
🧠 » Creating dense neural embeddings
🔍 » Building sparse token embeddings
🔎 » Running hybrid search - combining dense & sparse results
📚 » Found 5 most relevant chunks
🤖 » Querying LLM for answer generation
✅ » Answer generated successfully for question 1
🔎 » Running hybrid search - combining dense & sparse results
📚 » Found 5 most relevant chunks
🤖 » Querying LLM for answer generation
✅ » Answer generated successfully for question 2
⌛ » Total processing time 18.04s
```

### 🔄 Process Documents and Answer Questions
```http
POST /run
```

#### 📝 Request Body
```json
{
  "documents": [
    "https://arxiv.org/pdf/2005.11401.pdf",
    "https://arxiv.org/pdf/1706.03762.pdf"
  ],
  "questions": [
    "What are the two types of RAG models and how do they differ?",
    "What key innovation does the Transformer introduce over previous sequence models?"
  ]
}
```

#### Response
```json
{
  "answers": [
    "The two types of RAG models are RAG-Token and RAG-Sequence. They differ in their approach to generating text, with RAG-Token performing better on Jeopardy question generation and RAG-Sequence outperforming BART on Open MS-MARCO NLG.",
    "The Transformer introduces self-attention as its key innovation, replacing recurrent or convolutional layers used in previous sequence models with multi-headed self-attention."
  ]
}
```

## Error Responses

### 🚫 Authentication Error
```json
{
    "detail": "Invalid API key"
}
```

### ❌ Processing Error
```json
{
    "detail": "Error message describing what went wrong"
}
```

## 📦 Features

- Hybrid search combining dense and sparse embeddings
- Support for both local PDF files and remote URLs
- Automatic text chunking with overlap for context preservation
- Parallel processing of document embeddings
- Retry logic with exponential backoff for API calls
- Comprehensive error handling and logging

## 🤖 Technical Details

### Tech Stack

#### Backend Framework and API
- FastAPI - High-performance async web framework
- Uvicorn - ASGI server implementation
- CORS middleware for cross-origin support

#### Machine Learning and Embeddings
- Google's Gemini model (models/embedding-001) for document embeddings
- Groq LLM (llama-3.3-70b-versatile) for answer generation
- FAISS for efficient similarity search using cosine similarity
- TF-IDF vectorizer for sparse lexical matching
- NumPy for numerical operations and vector manipulations

#### Document Processing
- PyPDF2 for PDF text extraction
- Custom text chunking with configurable overlap
- Parallel processing using ThreadPoolExecutor and asyncio

#### Utilities and Error Handling
- Python-dotenv for environment management
- Tenacity for retry logic and backoff
- Pydantic for request/response validation
- Logging for production monitoring

#### RAG Pipeline Architecture

1. **Document Ingestion**
   - Supports both local PDF files and remote URLs
   - Extracts text content from PDF documents
   - Splits text into overlapping chunks (750 chars with 150 char overlap)

2. **Embedding Generation**
   - Processes text chunks in parallel using ThreadPoolExecutor
   - Generates dense embeddings using Google's Gemini model
   - Creates sparse embeddings using TF-IDF vectorization
   - Normalizes vectors using L2 normalization for cosine similarity

3. **Hybrid Search**
   - Combines dense and sparse search methods:
     - Dense: Semantic search using FAISS (Inner Product similarity)
     - Sparse: Lexical matching using TF-IDF
   - Weighted combination (default: 0.7 dense, 0.3 sparse)
   - Returns top K most relevant chunks (default: K=5)

4. **Answer Generation**
   - Takes user question and retrieved context chunks
   - Limited to 5 most relevant chunks for context window management
   - Uses Groq's LLM for answer generation
   - Implements retry logic with exponential backoff
   - Returns concise, plain text answers

5. **Performance Optimizations**
   - Parallel processing of document embeddings
   - Efficient vector similarity search with FAISS
   - Automatic chunking for optimal context windows
   - Request/response validation using Pydantic
   - Comprehensive error handling and logging

### ⚙️ System Configuration

```python
# Embedding Configuration
EMBED_DIM = 768  # Dimensionality of Gemini embeddings

# Text Chunking Parameters
CHUNK_SIZE = 750  # Characters per chunk
CHUNK_OVERLAP = 150  # Overlap between chunks

# Processing Parameters
BATCH_SIZE = 10  # Batch size for embedding generation
TOP_K = 5  # Number of relevant chunks to retrieve
MAX_DOCS = 5  # Maximum chunks in LLM context
```

## ⚡ Rate Limiting & Performance

- LLM requests include retry logic with exponential backoff
- Maximum of 3 retry attempts
- Backoff multiplier: 1 second
- Minimum wait time: 4 seconds
- Maximum wait time: 10 seconds
