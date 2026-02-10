"""
Main FastAPI application.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from pathlib import Path
import traceback
import sys
import re
import uuid
from datetime import datetime

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
from app.services.agent_service import StudyBuddyAgent

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

def get_agent_service(
    settings: Settings = Depends(get_settings),
    rag_service: RAGService = Depends(get_rag_service)
) -> StudyBuddyAgent:
    """Get agent service instance."""
    return StudyBuddyAgent(
        claude_api_key=settings.claude_api_key,
        rag_service=rag_service
    )

# Route Handlers
@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_document_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    rag_service: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings)
):
    """Upload and process a document."""
    try:
        # Validate file
        file_extension = Path(file.filename).suffix[1:].lower()
        
        if file_extension not in settings.allowed_file_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported: {file_extension}"
            )
        
        file_type = FileType(file_extension)
        
        # Read and validate size
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.max_file_size_bytes / (1024*1024)}MB"
            )
        
        # Save file
        file_path = await doc_service.save_file(file.filename, content)
        print(f"Saved file to: {file_path}")
        
        # Extract text
        text = doc_service.extract_text(file_path, file_type)
        print(f"Extracted {len(text)} characters")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = {
            "filename": file.filename,
            "file_type": file_type.value,
            "file_size_bytes": file_size,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        # Create chunks
        chunks = doc_service.create_chunks(
            text=text,
            document_id=document_id,
            metadata=metadata
        )
        
        print(f"Created {len(chunks)} chunks")
        
        # Embed chunks
        chunks_with_embeddings = await embedding_service.embed_chunks(chunks)
        print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
        
        # Store in vector database
        await rag_service.store_chunks(chunks_with_embeddings)
        print(f"Stored {len(chunks_with_embeddings)} chunks in vector DB")
        
        return UploadResponse(
            success=True, 
            message="Document uploaded successfully",
            document_id=document_id,
            filename=file.filename,
            chunks_created=len(chunks)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error uploading document: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
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
    
@app.post("/agent/query", response_model=QueryResponse)
async def agent_query(
    request: QueryRequest,
    agent: StudyBuddyAgent = Depends(get_agent_service)
):
    """
    Process query using AI agent (smarter than basic RAG).
    
    The agent can:
    - Handle multi-part questions
    - Generate practice questions
    - Perform comparisons
    - Break down complex queries
    """
    import time
    
    start_time = time.time()
    
    try:
        # Process with agent
        result = await agent.process_query(
            user_query=request.question,
            conversation_history=None  # Could add conversation memory here
        )
        
        query_time = time.time() - start_time
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query_time_seconds=round(query_time, 2)
        )
    
    except Exception as e:
        import traceback
        print(f"Agent error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )
    
@app.post("/preview", response_class=JSONResponse)
async def preview_document(
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings)
):
    """Preview extracted text and chunking from a document."""
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
        file_size = len(content)
        
        # Save file temporarily
        file_path = await doc_service.save_file(file.filename, content)
        
        print(f"Previewing file: {file.filename} ({file_size} bytes)")
        
        # Extract text
        extracted_text = doc_service.extract_text(file_path, file_type)
        
        # Calculate stats
        word_count = len(extracted_text.split())
        line_count = len(extracted_text.split('\n'))
        
        # Create complete metadata for chunks
        chunk_metadata = {
            "filename": file.filename,
            "file_type": file_type.value,        # ← Add this
            "file_size_bytes": file_size,        # ← Add this
            "upload_timestamp": datetime.now().isoformat()
        }
        
        # Create chunks for preview
        chunks = doc_service.create_chunks(
            text=extracted_text,
            document_id="preview",
            metadata=chunk_metadata              # ← Pass complete metadata
        )
        
        # Return preview with chunks
        return {
            "filename": file.filename,
            "file_type": file_type.value,
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 2),
            "extracted_length": len(extracted_text),
            "word_count": word_count,
            "line_count": line_count,
            "preview_first_500": extracted_text[:500],
            "preview_last_500": extracted_text[-500:] if len(extracted_text) > 500 else "",
            "full_text": extracted_text,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "length": len(chunk.text),
                    "word_count": len(chunk.text.split()),
                    "first_line": chunk.text.split('\n')[0][:100] if chunk.text else "",
                    "start_char": chunk.start_char,   # ← Include position info
                    "end_char": chunk.end_char,       # ← Include position info
                }
                for chunk in chunks
            ],
            "chunk_stats": {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
                "min_chunk_size": min(len(c.text) for c in chunks) if chunks else 0,
                "max_chunk_size": max(len(c.text) for c in chunks) if chunks else 0,
            }
        }
    
    except Exception as e:
        import traceback
        print(f"Error previewing document: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error previewing document: {str(e)}"
        )

    
@app.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "healthy"}

@app.get("/stats")
async def get_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get system statistics."""
    try:
        # Get collection info
        collection = rag_service.collection
        
        # Count total chunks
        total_chunks = collection.count()
        
        # Get unique document IDs
        if total_chunks > 0:
            results = collection.get()
            unique_docs = set()
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    if metadata and 'document_id' in metadata:
                        unique_docs.add(metadata['document_id'])
            total_documents = len(unique_docs)
        else:
            total_documents = 0
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_queries": rag_service.total_queries  # Make sure this is included
        }
    
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0
        }