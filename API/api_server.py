from fastapi import FastAPI, HTTPException, Security, Depends
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

# Load environment variables from .env file
load_dotenv()

# Configure application logging with INFO level for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================================================================
# Configuration Constants
# ========================================================================================

# Gemini embedding model configuration
GEMINI_MODEL = "models/embedding-001"
EMBED_DIM = 768  # Dimensionality of Gemini embeddings

# Groq LLM configuration for answer generation
LLM_MODEL = "llama-3.3-70b-versatile"

# Text chunking parameters for optimal context windows
CHUNK_SIZE = 750  # Size of text chunks in characters
CHUNK_OVERLAP = 150  # Overlap between chunks to maintain context

# Processing and retrieval parameters
BATCH_SIZE = 10  # Batch size for embedding generation
TOP_K = 5  # Number of relevant chunks to retrieve
MAX_DOCS = 5  # Maximum documents to include in LLM context

# Initialize FastAPI application with metadata
app = FastAPI(title="RAG API Server")

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

# Configure CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Overlapping chunks help maintain context across boundaries and improve
    retrieval quality for questions that span multiple sections.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between consecutive chunks
        
    Returns:
        List[str]: List of text chunks with specified overlap
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

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
    Process and index a document for retrieval.
    
    Performs the following operations:
    1. Chunks the text into overlapping segments
    2. Generates dense embeddings using Gemini
    3. Creates sparse TF-IDF representations
    4. Adds vectors to FAISS index with L2 normalization
    
    Args:
        text: The full document text to index
    """
    print("📝 » Extracting text content")
    print(f"✂️  » Text split into {len(chunk_text(text))} smart chunks")
    chunks = chunk_text(text)
    
    print("🧠 » Creating dense neural embeddings")
    # Dense embeddings
    embeddings = await embed_text_batch(chunks)
    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)  # Normalize for cosine similarity via inner product
    index.add(vectors)
    
    print("🔍 » Building sparse token embeddings")
    # Sparse embeddings
    global sparse_index, tfidf_vectorizer
    sparse_index = tfidf_vectorizer.fit_transform(chunks)
    
    # Store chunk metadata for retrieval
    for chunk in chunks:
        metadatas.append({"text": chunk})

async def hybrid_search(query: str, k: int = TOP_K, alpha: float = 0.7) -> List[str]:
    """
    Perform hybrid search combining dense neural and sparse lexical matching.
    
    Combines results from:
    1. Dense search: Semantic similarity using Gemini embeddings + FAISS
    2. Sparse search: Lexical matching using TF-IDF vectorization
    
    The hybrid approach improves recall by capturing both semantic meaning
    and exact keyword matches.
    
    Args:
        query: Search query string
        k: Number of results to return
        alpha: Weight for dense vs sparse results (0.0-1.0)
               1.0 = only dense, 0.0 = only sparse, 0.7 = balanced toward dense
        
    Returns:
        List[str]: List of relevant text chunks ordered by combined relevance score
    """
    start_time = time.time()
    print("🔎 » Running hybrid search - combining dense & sparse results")
    
    # Dense semantic search using FAISS
    q_emb = (await embed_text_batch([query]))[0]
    q_vec = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q_vec)
    D_dense, I_dense = index.search(q_vec, k)
    
    # Sparse lexical search using TF-IDF
    q_sparse = tfidf_vectorizer.transform([query])
    scores_sparse = (q_sparse * sparse_index.T).toarray()[0]
    I_sparse = np.argsort(scores_sparse)[::-1][:k]
    
    # Combine scores with weighted averaging
    combined_scores = {}
    
    # Add dense search results with alpha weighting
    for idx, score in zip(I_dense[0], D_dense[0]):
        if idx < len(metadatas):
            combined_scores[idx] = alpha * score
            
    # Add sparse search results with (1-alpha) weighting
    for idx, score in zip(I_sparse, scores_sparse[I_sparse]):
        if idx < len(metadatas):
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * score
    
    # Return top-k results sorted by combined relevance score
    top_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]
    results = [metadatas[i]["text"] for i in top_indices]
    
    duration = time.time() - start_time
    logger.info(f"Search took {duration:.2f}s")
    return results

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def ask_llm(query: str, context_chunks: List[str]) -> str:
    """
    Generate an answer using the LLM with retrieved context.
    
    Includes retry logic with exponential backoff for handling API rate limits
    and transient failures. Limits context to MAX_DOCS chunks to stay within
    token limits while providing sufficient information.
    
    Args:
        query: The user's question
        context_chunks: List of relevant text chunks from document retrieval
        
    Returns:
        str: Generated answer from the LLM
        
    Raises:
        Exception: After exhausting retry attempts
    """
    start_time = time.time()
    print(f"📚 » Found {len(context_chunks[:MAX_DOCS])} most relevant chunks")
    print("🤖 » Querying LLM for answer generation")
    
    # Prepare context by joining retrieved chunks
    context = "\n---\n".join(context_chunks[:MAX_DOCS])
    
    # Construct prompt with clear instructions for plain text output
    prompt = f"""You are a helpful assistant. Use the following context to answer the question in concise Return the answers in plain text only. Do not include any Markdown, special characters, escape sequences.

CONTEXT:
{context}

QUESTION:
{query}
"""
    
    # Generate response using Groq's LLM API
    resp = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,  # Limit response length for concise answers
    )
    
    duration = time.time() - start_time
    logger.info(f"LLM response took {duration:.2f}s")
    return resp.choices[0].message.content

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
            "/": {
                "method": "GET",
                "description": "Get API information"
            },
            "/run": {
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

@app.post("/run")
async def run_pipeline(request: RunRequest, api_key: str = Depends(verify_api_key)):
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
        if request.documents:
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
            for doc_url in request.documents:
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
        for idx, question in enumerate(request.questions, start=1):
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

if __name__ == "__main__":
    """
    Production server entry point for Render deployment.
    Uses PORT environment variable provided by Render.
    """
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)