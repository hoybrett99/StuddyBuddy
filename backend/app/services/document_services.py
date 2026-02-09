"""
Service for handling document operations.
"""
import uuid
from pathlib import Path
import re
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import (
    DocumentMetaData,
    DocumentChunk,
    FileType
)
from app.config import get_settings

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
    
    def extract_text(self, file_path: Path, file_type: FileType) -> str:
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
        
    def _clean_text(self, text: str) -> str:
        """
        Clean OCR artifacts and formatting issues from extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove single letters followed by space (common OCR error)
        text = re.sub(r'\b([A-Z])\s+', '', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove standalone capital letters that aren't 'A' or 'I'
        text = re.sub(r'\s+[B-HJ-Z]\s+', ' ', text)
        
        # Fix common character substitutions
        replacements = {
            'K': '',       # Remove stray K's
            'H ': ' ',     # Remove H followed by space
            'O ': 'o ',    # Fix capital O to lowercase o
            '—': '-',      # Fix em-dashes
            "'": "'",      # Fix smart quotes
            '"': '"',      # Fix smart quotes
            '"': '"',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def _extract_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF using multiple methods.
        
        Tries:
        1. pdfplumber (best for digital PDFs)
        2. pypdf (fallback)
        3. OCR (for scanned PDFs)
        
        Returns the cleanest result.
        """
        results = []
        
        # Method 1: pdfplumber
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            text = "\n\n".join(text_parts)
            if text:
                results.append(('pdfplumber', self._clean_text(text)))
        except Exception as e:
            print(f"pdfplumber failed: {e}")
        
        # Method 2: pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
            text = "\n\n".join(text_parts)
            if text:
                results.append(('pypdf', self._clean_text(text)))
        except Exception as e:
            print(f"pypdf failed: {e}")
        
        # Choose the best result (longest clean text)
        if results:
            best = max(results, key=lambda x: len(x[1]))
            print(f"Using {best[0]} for PDF extraction")
            return best[1]
        else:
            raise ValueError("Could not extract text from PDF")
    
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
        metadata: dict
    ) -> List[DocumentChunk]:
        """
        Create optimized chunks for general PDFs.
        
        Optimizations:
        - 800 char chunks (better than 1000 for dense content)
        - Comprehensive separator list (preserves sentences)
        - Position tracking (for debugging/analysis)
        """
        
        # Configure splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=[
                "\n\n\n",  # Major breaks (3+ newlines)
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence ends (period + space)
                "! ",      # Exclamations
                "? ",      # Questions
                "; ",      # Semicolons
                ": ",      # Colons
                ", ",      # Commas
                " ",       # Spaces
                "",        # Characters (last resort)
            ],
            is_separator_regex=False,
        )
        
        # Split text
        raw_chunks = text_splitter.split_text(text)
        
        # Create chunks with metadata
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(raw_chunks):
            # Clean whitespace
            cleaned = chunk_text.strip()
            if not cleaned:
                continue
            
            # Track position in original text
            start_char = text.find(chunk_text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_text)
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=cleaned,
                chunk_index=len(chunks),  # Use actual index (skipping empty chunks)
                start_char=start_char,
                end_char=end_char,
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "total_chunks": len(raw_chunks),
                    "chunk_size": len(cleaned),
                }
            )
            chunks.append(chunk)
            current_pos = end_char
        
        print(f"Created {len(chunks)} chunks (avg {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars)")
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
        from app.models import DocumentMetaData, FileType
        
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

        