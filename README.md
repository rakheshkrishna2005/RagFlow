# 🚀 RagFlow API + Client

RagFlow is a Retrieval Augmented Generation (RAG) API that processes PDF documents and answers questions using advanced language models.

## 📑 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [RAG Pipeline Architecture](#rag-pipeline-architecture)
- [System Configuration](#system-configuration)
- [Environment Variables](#environment-variables)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Error Responses](#error-responses)
- [Performance & Rate Limiting](#performance--rate-limiting)

## ✨ Features

- **Hybrid Search**: Combines dense (semantic) and sparse (lexical) embeddings for superior retrieval accuracy.
- **Flexible Document Loading**: Supports both local PDF files and remote URLs.
- **Advanced Text Chunking**: Uses `LangChain`'s `RecursiveCharacterTextSplitter` for intelligent context-aware chunking.
- **Optimized Performance**: Leverages parallel processing for embedding generation and includes caching for LLM and embedding lookups.
- **Resilient API Calls**: Implements retry logic with exponential backoff for external service calls.
- **Robust and Observable**: Features comprehensive error handling and structured logging for production monitoring.

## 🛠️ Tech Stack

| Category              | Tool / Library                                      |
|------------------------|-----------------------------------------------------|
| Backend               | FastAPI, Uvicorn                                   |
| Data Validation       | Pydantic                                           |
| Document Processing   | pypdf (for text extraction)                        |
| ML/RAG Orchestration  | LangChain (text splitting, prompts, LLM integration)|
| Embeddings            | Google Gemini (`models/embedding-001`)             |
| LLM                   | Groq via LangChain (`llama-3.3-70b-versatile`)     |
| Vector Search         | faiss (efficient similarity search)                 |
| Sparse Retrieval      | scikit-learn (TfidfVectorizer)                     |
| Utilities             | numpy, python-dotenv, tenacity                     |
| Containerization      | Docker                                             |
| CI/CD                 | GitHub Actions                                     |

## 🔄 RAG Pipeline Architecture

1.  **Document Ingestion**: Loads PDF documents from local paths or URLs and extracts text using `pypdf`.
2.  **Text Chunking**: Splits the extracted text into smaller, overlapping chunks using `LangChain`'s `RecursiveCharacterTextSplitter` to maintain semantic context.
3.  **Embedding Generation**: Creates two types of embeddings for hybrid search:
    -   **Dense Embeddings**: Generated using the Gemini model for capturing semantic meaning.
    -   **Sparse Embeddings**: Generated using TF-IDF for lexical matching.
4.  **Hybrid Search**: Combines dense and sparse search results with a weighted alpha to retrieve the most relevant text chunks for a given query.
5.  **Answer Generation**: Uses a `LangChain` prompt template to combine the user's question with the retrieved context. The `ChatGroq` LLM then generates a concise, plain-text answer.
6.  **Caching**: Implements `InMemoryCache` from `LangChain` for both LLM responses and embeddings to speed up repeated queries.

## ⚙️ System Configuration

```python
# Embedding Configuration
EMBED_DIM = 768        # Dimensionality of Gemini embeddings

# Text Chunking Parameters
CHUNK_SIZE = 750       # Characters per chunk
CHUNK_OVERLAP = 150    # Overlap between chunks

# Retrieval & Generation Parameters
TOP_K = 5              # Number of relevant chunks to retrieve
MAX_DOCS = 5           # Maximum chunks to include in LLM context
HYBRID_ALPHA = 0.7     # Weight for dense search in hybrid retrieval
```

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
API_KEY="your_secret_api_key"
GOOGLE_API_KEY="your_google_api_key"
GROQ_API_KEY="your_groq_api_key"
```

## 🔐 Authentication

All protected API endpoints require Bearer token authentication. Include the API key in the `Authorization` header:

```http
Authorization: Bearer your_secret_api_key
```

## 📡 API Endpoints

### `GET /`

Returns information about the API service, including its version and a description of available endpoints.

### `POST /run`

Processes one or more documents and answers a list of questions using the RAG pipeline.

#### Request Body

```json
{
  "documents": [
    "https://arxiv.org/pdf/2005.11401.pdf"
  ],
  "questions": [
    "What are the two types of RAG models and how do they differ?"
  ]
}
```

#### Response Body

```json
{
  "answers": [
    "The two types of RAG models are RAG-Token and RAG-Sequence. RAG-Token models treat each token as a latent variable and can be trained end-to-end, while RAG-Sequence models treat each document as a latent variable and use a more straightforward training process. RAG-Sequence models generally outperform RAG-Token models on tasks like open-domain question answering."
  ]
}
```

#### Using Curl

```bash
curl -X 'POST' \
  'https://rag-flow.onrender.com/v1/run' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer <api_key>' \
  -H 'Content-Type: application/json' \
  -d '{
  "documents": [
    "https://arxiv.org/pdf/2005.11401.pdf"
  ],
  "questions": [
    "What are the two types of RAG models and how do they differ?"
  ]
}'
```

#### Example Logs

```
🚀 » New request received from client
📑 » Loading PDF from web source: https://arxiv.org/pdf/2005.11401.pdf
✨ » PDF loaded successfully
📝 » Extracting text content
✂️ » Text split into 114 smart chunks
🔎 » Running hybrid search with LangChain EnsembleRetriever
📚 » Found 5 most relevant chunks
🤖 » Querying LLM for answer generation
✅ » Answer generated successfully for question 1
⌛ » Total processing time 8.45s
```

## ❌ Error Responses

-   **401 Unauthorized**: Returned if the API key is missing or invalid.
-   **500 Internal Server Error**: Returned for issues during document processing, embedding, or answer generation.

```json
{
    "detail": "Invalid API key"
}
```

## ⚡️ Performance & Rate Limiting

The API uses `tenacity` to implement an exponential backoff retry strategy for calls to the Groq LLM, making the system more resilient to transient failures.

-   **Max Retries**: 3
-   **Min Wait**: 4 seconds
-   **Max Wait**: 10 seconds
