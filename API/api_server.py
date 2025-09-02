from fastapi import APIRouter, FastAPI, HTTPException, Security, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import faiss
import numpy as np
import google.generativeai as genai
from groq import Groq
from pypdf import PdfReader
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from pydantic import BaseModel
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from io import BytesIO
import urllib.parse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.cache import InMemoryCache
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Configure application logging with INFO level for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================================================================
# Configuration Constants
# ========================================================================================

GEMINI_MODEL = "models/embedding-001"
EMBED_DIM = 768
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150
BATCH_SIZE = 10
TOP_K = 5
MAX_DOCS = 5

# LangChain cache setup
from langchain_community.cache import InMemoryCache
import langchain_core
langchain_core.llm_cache = InMemoryCache()
langchain_core.embeddings_cache = InMemoryCache()

# Rate limiting setup
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


# Initialize FastAPI application with metadata
app = FastAPI(title="RagFlow API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# HTTP Bearer token security scheme
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify the provided API key against the environment variable.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        str: The verified API key
        
    Raises:
        HTTPException: If the API key is invalid or missing
    """
    api_key = os.getenv("API_KEY")
    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return credentials.credentials

# API v1 router
v1_router = APIRouter(prefix="/v1")

# Configure CORS middleware for cross-origin requests

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Initialize external service clients
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize FAISS index with Inner Product similarity for normalized vectors
index = faiss.IndexFlatIP(EMBED_DIM)

# Global storage for document metadata and sparse search components
metadatas = []
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
sparse_index = None

# ========================================================================================
# Request/Response Models
# ========================================================================================

class RunRequest(BaseModel):
    """
    Request model for the RAG pipeline execution.
    
    Attributes:
        documents: List of URLs or file paths to PDF documents
        questions: List of questions to answer about the documents
    """
    documents: List[str]
    questions: List[str]

class SearchQuery(BaseModel):
    """
    Request model for semantic search queries.
    
    Attributes:
        query: The search query string
        k: Number of results to return (default: TOP_K)
        hybrid_alpha: Weight for combining dense and sparse search results (0-1)
    """
    query: str
    k: int = TOP_K
    hybrid_alpha: float = 0.7

# ========================================================================================
# Core Processing Functions
# ========================================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Use LangChain's RecursiveCharacterTextSplitter for better chunking.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

async def embed_text_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of text chunks using Google's Gemini model.
    
    Uses ThreadPoolExecutor to parallelize embedding generation for improved
    performance on multiple text chunks.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List[List[float]]: List of embedding vectors
        
    Raises:
        Exception: If embedding generation fails
    """
    start_time = time.time()
    try:
        embeddings = []

        def embed_one(text):
            """Single text embedding function for thread pool execution."""
            resp = genai.embed_content(model=GEMINI_MODEL, content=text)
            return resp["embedding"]

        # Execute embeddings in parallel using thread pool
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            embeddings = await asyncio.gather(
                *[loop.run_in_executor(pool, embed_one, text) for text in texts]
            )
        return embeddings
    except Exception as e:
        logger.error(f"Error in embed_text_batch: {str(e)}")
        raise

async def add_to_index(text: str):
    """
    Use LangChain for chunking and store chunks for retrieval.
    """
    print("📝 » Extracting text content")
    chunks = chunk_text(text)
    print(f"✂️ » Text split into {len(chunks)} smart chunks")
    # Store chunk metadata for retrieval
    for chunk in chunks:
        metadatas.append({"text": chunk})
    # Build dense and sparse indices for EnsembleRetriever
    global dense_embeddings, sparse_embeddings
    dense_embeddings = await embed_text_batch(chunks)
    sparse_embeddings = tfidf_vectorizer.fit_transform(chunks)

async def hybrid_search(query: str, k: int = TOP_K, alpha: float = 0.7) -> List[str]:
    """
    Use LangChain's EnsembleRetriever for hybrid search combining dense and sparse retrievers.
    """
    start_time = time.time()
    print("🔎 » Running hybrid search with LangChain EnsembleRetriever")
    
    # Get query embedding first
    query_embedding = (await embed_text_batch([query]))[0]
    
    # Dense retriever: cosine similarity on dense embeddings
    dense_scores = np.dot(dense_embeddings, np.array(query_embedding))
    dense_indices = np.argsort(dense_scores)[::-1][:k]
    
    # Sparse retriever: TF-IDF
    q_sparse = tfidf_vectorizer.transform([query])
    scores_sparse = (q_sparse * sparse_embeddings.T).toarray()[0]
    sparse_indices = np.argsort(scores_sparse)[::-1][:k]
    
    # Combine scores with weighted average
    combined_scores = {}
    for idx, score in zip(dense_indices, dense_scores[dense_indices]):
        combined_scores[idx] = alpha * score
    for idx, score in zip(sparse_indices, scores_sparse[sparse_indices]):
        combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * score
    
    # Get top-k results
    top_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]
    context_chunks = [metadatas[i]["text"] for i in top_indices]
    reranked_chunks = context_chunks
    
    duration = time.time() - start_time
    logger.info(f"Search took {duration:.2f}s")
    return reranked_chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def ask_llm(query: str, context_chunks: List[str]) -> str:
    """
    Use LangChain PromptTemplate and GroqLLM for answer generation, with cache.
    """
    start_time = time.time()
    print(f"📚 » Found {len(context_chunks[:MAX_DOCS])} most relevant chunks")
    print("🤖 » Querying LLM for answer generation")
    context = "\n---\n".join(context_chunks[:MAX_DOCS])
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. Use the following context to answer the question in concise Return the answers in plain text only. Do not include any Markdown, special characters, escape sequences.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"
    )
    prompt = prompt_template.format(context=context, question=query)
    llm = ChatGroq(model_name=LLM_MODEL)
    response = llm.invoke(prompt)
    answer = response.content
    duration = time.time() - start_time
    logger.info(f"LLM response took {duration:.2f}s")
    return answer

# ========================================================================================
# Error Handlers
# ========================================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom handler for HTTPException to ensure consistent JSON error responses.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Generic exception handler to catch any unhandled exceptions and return a
    standardized 500 Internal Server Error response.
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# ========================================================================================
# API Endpoints
# ========================================================================================

@app.get("/")
async def root():
    """
    API information endpoint providing service metadata and usage documentation.
    
    Returns:
        Dict: API service information including available endpoints and configuration
    """
    return {
        "name": "RAG API Server",
        "version": "1.0.0",
        "description": "A REST API for Question Answering using RAG (Retrieval Augmented Generation)",
        "endpoints": {
            "/v1/run": {
                "method": "POST",
                "description": "Process documents and answer questions using RAG",
                "authentication": "Bearer token required",
                "request_format": {
                    "documents": ["List of URLs or paths to PDF files"],
                    "questions": ["List of questions to answer about the documents"]
                },
                "response_format": {
                    "answers": ["List of answers corresponding to the questions"]
                }
            }
        },
        "components": {
            "embedding_model": GEMINI_MODEL,
            "llm_model": LLM_MODEL
        }
    }

@v1_router.post("/run")
@limiter.limit("5/minute")
async def run_pipeline(run_request: RunRequest, request: Request, api_key: str = Depends(verify_api_key)):
    """
    Main RAG pipeline endpoint for document processing and question answering.
    
    Process flow:
    1. Authenticate request using Bearer token
    2. Clear existing index state for new document
    3. Load PDF from URL or local file path
    4. Extract and chunk text content
    5. Generate embeddings and build search indices
    6. Process each question through hybrid retrieval + LLM generation
    
    Args:
        request: RunRequest containing document path/URL and questions list
        api_key: Verified API key from dependency injection
        
    Returns:
        Dict: Response containing list of answers corresponding to input questions
        
    Raises:
        HTTPException: For authentication, file access, or processing errors
    """
    try:
        # Log request initiation for production monitoring
        if run_request.documents:
            start_time = time.time()
            print("\n🚀 » New request received from client")
        
        # Reset global state for new document processing
        global index, metadatas, sparse_index
        index = faiss.IndexFlatIP(EMBED_DIM)
        metadatas = []
        sparse_index = None

        # Document loading and text extraction
        try:
            full_text = ""
            for doc_url in run_request.documents:
                # Handle remote PDF URLs
                if doc_url.startswith(('http://', 'https://')):
                    print(f"📑 » Loading PDF from web source: {doc_url}")
                    response = requests.get(doc_url)
                    response.raise_for_status()
                    pdf = PdfReader(BytesIO(response.content))
                    print("✨ » PDF loaded successfully")
                else:
                    # Handle local file paths
                    print(f"\nReading local PDF: {doc_url}")
                    pdf = PdfReader(doc_url)
                
                # Extract text content from all PDF pages
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n\n--- New Document ---\n\n"
            
            # Process combined documents and build search indices
            if full_text:
                await add_to_index(full_text)
            
        except requests.RequestException as e:
            logger.error(f"PDF download failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error downloading PDF: {str(e)}")
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # Process questions through RAG pipeline
        answers = []
        for idx, question in enumerate(run_request.questions, start=1):
            # Retrieve relevant context using hybrid search
            retrieved_chunks = await hybrid_search(question, TOP_K)
            # Generate answer using LLM with retrieved context
            answer = await ask_llm(question, retrieved_chunks)
            answers.append(answer)
            print(f"✅ » Answer generated successfully for question {idx}")
        # Log total processing time for performance monitoring
        if 'start_time' in locals():
            total_time = time.time() - start_time
            print(f"⌛ » Total processing time {total_time:.2f}s")
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================================================================
# Application Entry Point
# ========================================================================================

app.include_router(v1_router)

if __name__ == "__main__":
    """
    Production server entry point for Render deployment.
    Uses PORT environment variable provided by Render.
    """
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
