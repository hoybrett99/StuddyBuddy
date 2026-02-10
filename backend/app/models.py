from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class FileType(str, Enum):
    """
    This prevents typos and makes code safer.
    """
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"

class DocumentMetaData(BaseModel):
    """
    Metadata about an uploaded document.
    """
    filename: str
    file_type: FileType
    upload_date: datetime = Field(default_factory=datetime.now)

    total_chunks: int = 0 #starts at 0
    file_size_bytes: int

    class ConfigDict:
        extra = "ignore"

class UploadResponse(BaseModel):
    """
    What we send back to the user after they upload their files.
    """
    success: bool = True
    message: str
    chunks_created: int
    document_id: str
    filename: str

class QueryRequest(BaseModel):
    """
    What the user sends when asking questions
    """
    question: str = Field(
                        ..., # required
                        min_length=1, # at least one character
                        max_length=100, # max 100 characters
                        description="The question to ask about the uploaded documents"
                        )
    document_ids: Optional[List[str]] = None # limit search to specific documents
    num_contexts: int = Field(default=4, ge=1, le=10)
    conversation_history: Optional[List[Dict[str, str]]] = None

    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v=str) -> str:
        """
        Custom validation runs automatically
        """
        # Strip whitespace
        v = v.strip()

        # Check if emptty after stripping
        if not v:
            raise ValueError("Question cannot be empty or whitespace")
        return v

class Source(BaseModel):
    """
    A source reference in the answer
    """
    document_name: str
    chunk_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    chunk_text: str = ""

class QueryResponse(BaseModel):
    """
    What we send back after answering a question
    """
    answer: str
    sources: List[Source] # list of source object
    query_time_seconds: float

class DocumentChunk(BaseModel):
    """A chunk of text from a document."""
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: Optional[int] = 0  # ← Make optional with default
    end_char: Optional[int] = 0    # ← Make optional with default
    metadata: Dict[str, Any] = Field(default_factory=dict)  # ← Make flexible
    embedding: Optional[list] = None

class ErrorResponse(BaseModel):
    """
    Standardised error response
    """
    error: bool=True
    message: str
    error_type: str
    details: Optional[dict] = None

class SystemStats(BaseModel):
    """
    Statistics about the system
    """
    total_documents: int
    total_chunks: int
    total_queries_processed: int
    vector_db_status: str
    last_upload: Optional[datetime] = None