"""
Main FastAPI application.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import time
from pathlib import Path
import traceback
import sys

from app.models import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    ErrorResponse,
    SystemStats,
    FileType,
    DocumentMetaData
)
from app.config import Settings, get_settings
from app.services.document_services import DocumentService
from app.services.embedding_services import EmbeddingService
from app.services.rag_services import RAGService

# FastAPI App
app = FastAPI(
    title="Study Buddy API",
    description="RAG based study assistant",
    version="0.0.1"
)

# Allows frontend to call the api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# These functions create instances of our services
# FastAPI will call them automatically when needed

def get_document_service() -> DocumentService:
    return DocumentService()

def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

def get_rag_service() -> RAGService:
    return RAGService()

# Route Handlers
@app.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document",
    description="Upload a PDF, TXT, or DOCX file for processing"
)
async def upload_document(
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_document_service),
    emb_service: EmbeddingService = Depends(get_embedding_service),
    rag_service: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings)
):
    """
    Upload and process a document.
    
    The function signature tells FastAPI:
    - Expect a file upload
    - Inject the services we need
    - Return an UploadResponse
    """
    try:
        # Validate file type
        file_extension = Path(file.filename).suffix[1:].lower()

        if file_extension not in settings.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type .{file_extension} not supported"
            )
        
        file_type = FileType(file_extension)

        # Read file content
        content = await file.read()

        # Check file size
        file_size = len(content)
        max_size = settings.max_file_size_mb * 1024 * 1024

        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File is currently {file_size} bytes. Max size: {settings.max_file_size_mb}MB."
            )
        
        # Save file
        file_path = await doc_service.save_file(file.filename, content)

        # Extract text
        text = doc_service.extract_text(file_path, file_type)

        # Create metadata
        document_id = file_path.stem # filename without extension
        metadata = DocumentMetaData(
            filename=file.filename,
            file_type=file_type,
            file_size_bytes=file_size
        )

        # Create chunks
        chunks = doc_service.create_chunks(text, document_id, metadata)

        # Generate embeddings
        chunks_with_embeddings = await emb_service.embed_chunks(chunks)

        # Store in chromaDB
        await rag_service.store_chunks(chunks_with_embeddings)

        # Updating metadata
        metadata.total_chunks = len(chunks)

        # return response
        return UploadResponse(
            success=True,
            message="Document processed successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            document_id=document_id,
        )
    
    except HTTPException:
            raise

    except Exception as e:
        # Print full traceback to console
        print("\n" + "="*80)
        print("ERROR IN UPLOAD:")
        print("="*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc(file=sys.stdout)
        print("="*80 + "\n")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {type(e).__name__}: {str(e)}"
        )

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description="Query the RAG system with a question"
)
async def query(
        request: QueryRequest,
        rag_service: RAGService = Depends(get_rag_service)
):
    """
    Ask a question about uploaded documents.
    
    FastAPI automatically:
    1. Parses JSON from request body
    2. Validates it against QueryRequest model
    3. Gives us a QueryRequest object
    4. Returns validation errors if invalid
    """
    start_time = time.time()

    try:
        # Get answer from RAG service
        answer, sources = await rag_service.query(
            question=request.question,
            num_contexts=request.num_contexts,
            document_ids=request.document_ids
        )

        query_time = time.time() - start_time

        return QueryResponse(
            answer=answer,
            sources=sources,
            query_time_seconds=round(query_time, 2) 
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
    
@app.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "healthy"}

@app.get("/stats", response_model=SystemStats)
async def get_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get system statistics."""
    stats = await rag_service.get_stats()
    return stats