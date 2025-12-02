# vector_search/embeddings.py
import numpy as np
from typing import List
import re


class SimpleEmbeddingGenerator:
    """A simple embedding generator for demonstration purposes"""

    def __init__(self, dimension: int = 300):
        self.dimension = dimension
        self.vocab = {}
        self.vocab_size = 0

    def build_vocab(self, documents: List[str]):
        """Build vocabulary from documents"""
        words = set()
        for doc in documents:
            # Simple tokenization
            tokens = re.findall(r'\b\w+\b', doc.lower())
            words.update(tokens)

        self.vocab = {word: i for i, word in enumerate(sorted(words))}
        self.vocab_size = len(self.vocab)

        # Initialize random embedding matrix
        np.random.seed(42)
        self.embedding_matrix = np.random.randn(self.vocab_size, self.dimension)

    def document_to_vector(self, document: str) -> np.ndarray:
        """Convert document to vector using average of word embeddings"""
        tokens = re.findall(r'\b\w+\b', document.lower())

        if not tokens:
            return np.zeros(self.dimension)

        vectors = []
        for token in tokens:
            if token in self.vocab:
                idx = self.vocab[token]
                vectors.append(self.embedding_matrix[idx])

        if not vectors:
            return np.zeros(self.dimension)

        # Return average vector
        return np.mean(vectors, axis=0)

    def documents_to_vectors(self, documents: List[str]) -> np.ndarray:
        """Convert multiple documents to vectors"""
        vectors = []
        for doc in documents:
            vectors.append(self.document_to_vector(doc))
        return np.array(vectors)