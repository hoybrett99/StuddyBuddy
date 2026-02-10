"""
Service for generating embeddings from text.
Embeddings are numerical representations that capture semantic meaning.
"""

from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

from app.models import DocumentChunk
from app.config import get_settings

class EmbeddingService:
    """
    Handles text embedding generation.
    
    Uses sentence-transformers library which runs locally.
    """
    def __init__(self):
        """
        Initialize the embedding model.
        
        This downloads the model on first run (~100MB).
        Subsequent runs load it from cache.
        """
        self.settings = get_settings()

        # Load the model
        self.model = SentenceTransformer(self.settings.embedding_model)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    # Embedding a single text
    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text string into an embedding vector.
        
        Args:
            text: The text to embed
            
        Returns:
            List[float]: A vector of numbers representing the text
            
        Example:
            >>> service = EmbeddingService()
            >>> vector = service.embed_text("What is photosynthesis?")
            >>> len(vector)
            384
            >>> vector[:3]
            [0.234, -0.123, 0.567]
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True # for better similarity search
        )

        # Covnert numpy array to list (for JSON serialisation)
        return embedding.tolist()
    
    # Embedding multiple texts (Batch Processing)
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts into embeddings efficiently.
        
        This is MUCH faster than calling embed_text() in a loop
        because the model can process multiple texts in parallel.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
            
        Example:
            >>> texts = ["Hello world", "Goodbye world", "Python is fun"]
            >>> vectors = service.embed_texts(texts)
            >>> len(vectors)
            3
            >>> len(vectors[0])
            384
        """
        if not texts:
            return []
        
        print(f"Embedding {len(texts)} text chunks...")

        # Batch encoding is much faster!
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,  # Show progress for large batches
            batch_size=32  # Process 32 texts at a time
        )
        
        # Convert to list of lists
        return embeddings.tolist()

    # Embed document chunks
    async def embed_chunks(
            self,
            chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        Add embeddings to DocumentChunk objects.
        
        This takes chunks WITHOUT embeddings and returns chunks WITH embeddings.
        
        Args:
            chunks: List of DocumentChunk objects (embedding field is None)
            
        Returns:
            List of DocumentChunk objects with embeddings filled in
            
        Example:
            Input chunk:  DocumentChunk(text="Hello", embedding=None)
            Output chunk: DocumentChunk(text="Hello", embedding=[0.234, -0.123, ...])
        """
        if not chunks:
            return []
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
    
        batch_size = 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            texts = [chunk.text for chunk in batch]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding.tolist()
        
        print(f"All embeddings generated!")
        return chunks
    
    # Calculate similarity between two texts
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate how similar two texts are (0 to 1).
        
        Uses cosine similarity:
        - 1.0 = identical meaning
        - 0.0 = completely different
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
            
        Example:
            >>> service.calculate_similarity("cat", "kitten")
            0.87
            >>> service.calculate_similarity("cat", "car")
            0.23
        """
        # Getting embeddings
        emb1 = np.array(self.embed_text(text1))
        emb2 = np.array(self.embed_text(text2))

        # Cosine similarity
        similarity = np.dot(emb1, emb2)

        return float(similarity)
    
if __name__ == "__main__":
    """
    Test the embedding service.
    """
    print("Testing EmbeddingService...\n")

    service = EmbeddingService()

    # Test 1: Single text embedding
    print("\n1. Testing single text embedding:")
    text = "Photosynthesis is the process by which plants make food."
    text2 = "Mitochondria is the powerhouse of the cell."
    embedding = service.embed_text(text)
    embedding2 = service.embed_text(text2)
    print(f"   Text: '{text}'")
    print(f"   Embedding length: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")

    print(f"   Text: '{text2}'")
    print(f"   Embedding length: {len(embedding2)}")
    print(f"   First 5 values: {embedding2[:5]}")
    
    # Test 2: Batch embedding
    print("\n2. Testing batch embedding:")
    texts = [
        "Plants use sunlight to make energy",
        "Animals eat food for energy",
        "Python is a programming language"
    ]
    embeddings = service.embed_texts(texts)
    print(f"   Embedded {len(embeddings)} texts")
    
    # Test 3: Similarity
    print("\n3. Testing similarity:")
    sim1 = service.calculate_similarity(texts[0], texts[1])
    sim2 = service.calculate_similarity(texts[0], texts[2])
    print(f"   '{texts[0]}' vs '{texts[1]}': {sim1:.3f}")
    print(f"   '{texts[0]}' vs '{texts[2]}': {sim2:.3f}")
    print(f"   (Higher = more similar)")
    
    print("\n✓ All tests passed!")