"""
RAG (Retrieval-Augmented Generation) Service.
This is the brain of the Study Buddy - it retrieves relevant context
and generates answers using an LLM.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Tuple, Optional
import google.generativeai as genai
from anthropic import Anthropic
from datetime import datetime

from app.models import (
    DocumentChunk,
    Source,
    SystemStats
)
from app.config import get_settings
from app.services.embedding_services import EmbeddingService

class RAGService:
    """
    Handles the full RAG pipeline:
    1. Store document chunks in vector database
    2. Search for relevant chunks given a query
    3. Generate answers using retrieved context
    """

    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()

        # Initialising ChromaDB
        print("Connecting to ChromaDB...")

        # Local development
        self.chroma_client = chromadb.Client(
            ChromaSettings(
                persist_directory="./chroma_data",  # Where to save data
                anonymized_telemetry=False
            )
        )

        # For Docker (uncomment when using docker-compose)
        # self.chroma_client = chromadb.HttpClient(
        #     host=self.settings.chroma_host,
        #     port=self.settings.chroma_port
        # )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name = self.settings.chroma_collection,
            metadata = {"description": "Study Buddy document chunks"}
        )

        print(f"Connected to collection '{self.settings.chroma_collection}'")

        # Initialising LLM (Claude)
        print("Initializing Claude...")
        self.client = Anthropic(api_key=self.settings.claude_api_key)
        print(f"Using model: {self.settings.claude_model}")

        # Stats tracking
        self.total_queries = 0

    # Storing chunks in vector database
    async def store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Store document chunks with embeddings in ChromaDB.
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
        """
        if not chunks:
            print("No chunks to store")
            return
        
        print(f"Storing {len(chunks)} chunks in vector database...")
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            
            # Prepare metadatas - use dictionary access
            metadatas = []
            for chunk in chunks:
                metadata = {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "filename": chunk.metadata.get("filename", "unknown"),
                    "file_type": chunk.metadata.get("file_type", "unknown"),
                    "chunk_size": chunk.metadata.get("chunk_size", len(chunk.text)),
                    "upload_timestamp": chunk.metadata.get("upload_timestamp", ""),
                }
                metadatas.append(metadata)
            
            # Validate embeddings
            if any(emb is None for emb in embeddings):
                raise ValueError("Some chunks are missing embeddings")
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            print(f"✅ Successfully stored {len(chunks)} chunks")
            
        except Exception as e:
            print(f"Error storing chunks: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    # Searching for relevant chunks
    async def search_similar_chunks(
            self,
            query: str,
            num_results: int=4,
            document_ids: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Find the most relevant chunks for a query.
        
        Args:
            query: The user's question
            num_results: How many chunks to return
            document_ids: Optional list to filter by specific documents
            
        Returns:
            List of (chunk, similarity_score) tuples
            
        How it works:
            1. Embed the query using the same model
            2. ChromaDB finds chunks with similar embeddings
            3. Return top-k most similar chunks
        """
        # Generate embedding for query
        query_embedding = self.embedding_service.embed_text(query)

        # Build filter if document_ids provided
        where_filter = None
        if document_ids:
            where_filter = {"document_id": {"$in": document_ids}}

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=num_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Convert results to DocumentChunk objects with scores
        chunks_with_scores = []

        # ChromaDB returns results as lists of lists
        for i in range(len(results['ids'][0])):
            # Extract data
            chunk_id = results['ids'][0][i]
            text = results['documents'][0][i]
            metadata_dict = results['metadatas'][0][i]
            distance = results['distances'][0][i]

            # Convert distance to similarity score (0-1)
            # ChromaDB uses L2 distance, convert to similarity
            similarity = 1 / (1 + distance)

            # We don't reconstruct full DocumentChunk here
            # Just store what we need for the response
            chunk_info = {
                'chunk_id': chunk_id,
                'text': text,
                'document_id': metadata_dict['document_id'],
                'filename': metadata_dict['filename'],
                'similarity': similarity
            }
            
            chunks_with_scores.append(chunk_info)
        
        return chunks_with_scores
    
    # Full RAG pipeline
    async def query(
            self,
            question: str,
            num_contexts: int = 4,
            document_ids: Optional[List[str]] = None
    ) -> Tuple[str, List[Source]]:
        """
        Answer a question using RAG.
        
        This is the MAIN method that ties everything together!
        
        Args:
            question: The user's question
            num_contexts: How many context chunks to use
            document_ids: Optional filter for specific documents
            
        Returns:
            Tuple of (answer, sources)
            
        The RAG Pipeline:
            1. Search for relevant chunks (RETRIEVAL)
            2. Build prompt with context
            3. Call LLM to generate answer (GENERATION)
            4. Return answer with sources
        """
        self.total_queries += 1

        # Step 1 (Retrieval)- Finding relevant chunks
        print(f"\nProcessing query: '{question}'")
        print(f"Searching for {num_contexts} relevant chunks...")

        chunks_with_scores = await self.search_similar_chunks(
            query=question,
            num_results=num_contexts,
            document_ids=document_ids
        )

        if not chunks_with_scores:
            return (
                "I couldn't find any relevant information in your uploaded documents to answer this question.",
                []
            )
        
        # Step 2 (Context)- Build context from retrieved chunks
        context_parts = []
        sources = []

        for idx, chunk_info in enumerate(chunks_with_scores, 1):
            # Add to context
            context_parts.append(
                f"[Source {idx} - {chunk_info['filename']}]\n{chunk_info['text']}"
            )

            # Create source object
            source = Source(
                document_name=chunk_info["filename"],
                chunk_id=chunk_info['chunk_id'],
                relevance_score=round(chunk_info['similarity'], 3),
                chunk_text=chunk_info['text'] 
            )
            sources.append(source)

        # Combine all context
        full_context = "\n\n".join(context_parts)
        
        print(f"Found {len(chunks_with_scores)} relevant chunks")
        print(f"Top relevance score: {sources[0].relevance_score}")

        # Step 3 (Generation)- Build prompt and call LLM
        print("Generating answer with Claude...")
        
        try:
            # Claude uses a messages API:
            # - system: Instructions for Claude's behavior
            # - messages: The conversation (user and assistant turns)
            
            response = self.client.messages.create(
                model=self.settings.claude_model,
                max_tokens=self.settings.claude_max_tokens,
                
                # System prompt - sets Claude's role and behavior
                system=self._get_system_prompt(),
                
                # User message with context and question
                messages=[
                    {
                        "role": "user",
                        "content": self._build_user_message(question, full_context)
                    }
                ]
            )
            
            # Extract the text from Claude's response
            answer = response.content[0].text
            print(f"Answer generated ({len(answer)} characters)")
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = f"Sorry, I encountered an error generating the answer: {str(e)}"
        
        return answer, sources
    
    # Prompt Engineering
    def _get_system_prompt(self) -> str:
        """
        System prompt defines Claude's role and behavior.
        
        This is separate from the user message in Claude's API.
        Think of it as "configuring" Claude before the conversation.
        """
        return """You are a helpful study assistant helping students understand their course materials.

Your role and responsibilities:
- Answer questions based ONLY on the provided context from the student's documents
- Be clear, accurate, and educational in your explanations
- Break down complex topics into understandable parts
- Use examples when helpful
- If the context doesn't contain enough information to fully answer the question, be honest about this
- Cite sources when using specific information (e.g., "According to Source 1...")

Your tone should be:
- Patient and encouraging
- Clear and concise
- Appropriate for students
- Professional but friendly

Remember: Base all answers on the provided context. Do not use outside knowledge unless the context is insufficient, in which case you should clearly state this."""
        
        return prompt
    
    def _build_user_message(self, question: str, context: str) -> str:
        """
        Build the user message with context and question.
        
        Claude performs best with clear structure and explicit instructions.
        """
        return f"""Here is the context from the student's uploaded study materials:

<context>
{context}
</context>

The student's question is: {question}

Please answer the question based on the context provided above. If you reference specific information, mention which source it came from. If the context doesn't fully answer the question, explain what information is available and what's missing."""

    
    # Getting system statistics
    async def get_stats(self)-> SystemStats:
        """
        Get statistics about the RAG system.
        
        Returns:
            SystemStats: System statistics
        """
        # Get count from chromaDB
        total_chunks = self.collection.count()

        # Count unique documents
        if total_chunks > 0:
            all_metadata = self.collection.get(include=["metadatas"])
            unique_docs = set(
                meta['document_id'] 
                for meta in all_metadata['metadatas']
            )
            total_documents = len(unique_docs)
        else:
            total_documents = 0
        
        return SystemStats(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_queries_processed=self.total_queries,
            vector_db_status="connected",
            last_upload=datetime.now() if total_chunks > 0 else None
        )
    
    # Clean-up Method of ChromaDB
    async def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            int: Number of chunks deleted
        """
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )

        chunk_ids = results['ids']

        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
            print(f"Deleted {len(chunk_ids)} chunks for document {document_id}")
        
        return len(chunk_ids)
    

# Testing the RAG Service
if __name__ == "__main__":
    """
    Test the RAG service with Claude.
    Run: python -m app.services.rag_service
    """
    import asyncio
    from app.models import DocumentMetaData, FileType
    
    async def test_rag():
        print("Testing RAGService with Claude...\n")
        
        # Create service
        service = RAGService()
        
        # Test 1: Create and store sample chunks
        print("\n1. Creating sample document chunks:")
        
        metadata = DocumentMetaData(
            filename="biology_test.txt",
            file_type=FileType.TXT,
            file_size_bytes=1000
        )
        
        # Sample text about photosynthesis
        sample_texts = [
            "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in the chloroplasts of plant cells.",
            "During photosynthesis, plants take in carbon dioxide and water, and using sunlight, produce glucose and oxygen.",
            "The overall equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2",
            "Chlorophyll is the green pigment in plants that absorbs light energy during photosynthesis."
        ]
        
        # Create chunks with embeddings
        embedding_service = EmbeddingService()
        chunks = []
        
        for i, text in enumerate(sample_texts):
            embedding = embedding_service.embed_text(text)
            chunk = DocumentChunk(
                chunk_id=f"test_doc_chunk_{i}",
                document_id="test_doc",
                text=text,
                chunk_index=i,
                start_char=i * 100,
                end_char=(i + 1) * 100,
                metadata=metadata,
                embedding=embedding
            )
            chunks.append(chunk)
        
        # Store chunks
        await service.store_chunks(chunks)
        
        # Test 2: Query the system
        print("\n2. Testing RAG query with Claude:")
        question = "What is photosynthesis and what is the chemical equation?"
        
        answer, sources = await service.query(question, num_contexts=3)
        
        print(f"\nQuestion: {question}")
        print(f"\nAnswer from Claude:\n{answer}")
        print(f"\nSources ({len(sources)}):")
        for source in sources:
            print(f"  - {source.document_name} (relevance: {source.relevance_score})")
        
        # Test 3: Get stats
        print("\n3. System statistics:")
        stats = await service.get_stats()
        print(f"  Total documents: {stats.total_documents}")
        print(f"  Total chunks: {stats.total_chunks}")
        print(f"  Queries processed: {stats.total_queries_processed}")
        
        print("\n✓ All tests passed!")
    
    # Run async test
    asyncio.run(test_rag())