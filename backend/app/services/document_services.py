"""
Service for handling document operations.
"""
import uuid
from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..models import (
    DocumentMetaData,
    DocumentChunk,
    FileType
)
from ..config import get_settings

class DocumentService:
    """
    A class groups related functions (methods) together.
    
    'self' refers to the instance of the class.
    Think of self as "this specific DocumentService object"
    """
    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.settings.chunk_size,
            chunk_overlap = self.settings.chunk_overlap,
            length_function = len
        )

    # Async functions
    async def save_file(self, filename: str, content: bytes) -> Path:
                """
                Save uploaded file to disk.
                
                Args:
                    filename: Name of the file
                    content: The file's bytes
                    
                Returns:
                    Path: Where the file was saved
                """
                # Create unique filenames to avoid conflicts
                file_id = uuid.uuid4().hex[:8]
                safe_filename = f"{file_id}_{filename}"
                file_path = self.upload_dir / safe_filename

                with open(file_path, 'wb') as f:
                        f.write(content)

                        return file_path
    
    def text_extraction(self, file_path: Path, file_type: FileType) -> str:
            """
            Extract text from a document.
            
            Args:
                file_path: Path to the file
                file_type: Type of file (PDF, TXT, DOCX)
                
            Returns:
                str: Extracted text
                
            Raises:
                ValueError: If file type not supported
            """
            if file_type == FileType.PDF:
                   return self._extract_pdf(file_path)
            elif file_type == FileType.TXT:
                return self._extract_txt(file_path)
            elif file_type == FileType.DOCX:
                return self._extract_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        
    def _extract_pdf(self, file_path: Path) -> str:
        """
        Extract text content from a PDF file.
        
        The underscore prefix (_extract_pdf) indicates this is a private/internal method.
        It should only be called from within the DocumentService class.
        
        Args:
            file_path (Path): Path to the PDF file to extract text from
            
        Returns:
            str: Extracted text from all pages, with pages separated by double newlines
            
        Raises:
            PdfReadError: If the PDF file is corrupted or cannot be read
            FileNotFoundError: If the file doesn't exist
            
        Example:
            >>> file_path = Path("/uploads/document.pdf")
            >>> text = self._extract_pdf(file_path)
            >>> print(text[:100])  # First 100 characters
            "Chapter 1: Introduction
            
            This is the beginning of the document..."
        """
        # Create a PDF reader object for this file
        reader = PdfReader(file_path)
        
        # List to store text from each page
        text_parts = []
        
        # Loop through each page in the PDF
        for page in reader.pages:
            # Extract text from this page
            text = page.extract_text()
            
            # Only add non-empty text
            if text:
                text_parts.append(text)  # Fixed: added text parameter
        
        # Join all pages with double newlines between them
        return "\n\n".join(text_parts)
    
    def _extract_txt(self, file_path: Path) -> str:
         """Extract text from txt files"""
         with open (file_path, 'r', encoding='utf-8') as f:
              return f.read()
         
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        from docx import Document
        doc = Document(file_path)
        return "\n\n".join([para.text for para in doc.paragraphs])
        
    def create_chunks(
              self,
              text: str,
              document_id: str,
              metadata: DocumentMetaData
    ) -> List[DocumentChunk]:
         """
         Split texts into chunks

         Returns:
            List[DocumentChunk]: List of chunk objects
         """
         # Split text
         text_chunks = self.text_splitter.split_text(text)

         # curernt document chunks
         chunks = []
         current_pos = 0

         for idx, chunk_text in enumerate(text_chunks):
              chunk = DocumentChunk(
                   chunk_id=f"{document_id}_chunk_{idx}",
                   document_id=document_id,
                   text=chunk_text,
                   chunk_index=idx,
                   start_char=current_pos,
                   end_char=current_pos + len(chunk_text),
                   metadata=metadata
              )
              chunks.append(chunk)
              current_pos += len(chunk_text)
         return chunks

if __name__ == "__main__":
    """
    Manual testing with sample documents.
    Run: python -m backend.app.services.document_services
    """
    try:
        print("=" * 60)
        print("DocumentService Manual Test with Sample Documents")
        print("=" * 60)
        
        # Initialize service
        doc1 = DocumentService()
        print("✓ DocumentService initialized\n")
        
        # ==============================================================
        # Test 1: Create chunks from sample text
        # ==============================================================
        print("Test 1: Text Chunking")
        print("-" * 60)
        
        # Sample document text
        sample_text = """
        Introduction to Biology
        
        Biology is the natural science that studies life and living organisms.
        This includes their physical structure, chemical processes, molecular 
        interactions, physiological mechanisms, development and evolution.
        
        Chapter 1: Cell Structure
        
        Cells are the basic building blocks of all living things. The human body 
        is composed of trillions of cells. They provide structure for the body, 
        take in nutrients from food, convert those nutrients into energy, and 
        carry out specialized functions.
        
        Chapter 2: Genetics
        
        Genetics is a branch of biology concerned with the study of genes, 
        genetic variation, and heredity in organisms. Gregor Mendel, a scientist 
        and Augustinian friar, gained posthumous recognition as the founder of 
        the modern science of genetics.
        """ * 10  # Repeat to make it longer for multiple chunks
        
        # Create metadata
        from backend.app.models import DocumentMetaData, FileType
        
        metadata = DocumentMetaData(
            filename="biology_textbook.pdf",
            file_type=FileType.PDF,
            file_size_bytes=len(sample_text.encode('utf-8'))
        )
        
        # Create chunks
        document_id = "doc_biology_001"
        chunks = doc1.create_chunks(sample_text, document_id, metadata)
        
        print(f"✓ Created {len(chunks)} chunks")
        print(f"✓ First chunk ID: {chunks[0].chunk_id}")
        print(f"✓ First chunk preview: {chunks[0].text[:100]}...")
        print(f"✓ Chunk positions: start={chunks[0].start_char}, end={chunks[0].end_char}")
        
        # ==============================================================
        # Test 2: Display all chunks summary
        # ==============================================================
        print("\nTest 2: Chunks Summary")
        print("-" * 60)
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Length: {len(chunk.text)} chars")
            print(f"  Position: {chunk.start_char} -> {chunk.end_char}")
            print(f"  Preview: {chunk.text[:80].strip()}...")
            print()
        
        print("=" * 60)
        print("All tests completed successfully! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

        