# vector_search/search_engine.py
import numpy as np
from typing import List, Dict, Tuple, Any
import json
import os


class VectorSearchEngine:
    """A vector search engine demonstrating linear algebra concepts"""

    def __init__(self, dimension: int = 300):
        self.dimension = dimension
        self.documents: List[str] = []
        self.vectors: np.ndarray = None
        self.document_metadata: List[Dict] = []
        self.index_built = False

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the search engine"""
        self.documents.extend(documents)

        if metadata:
            self.document_metadata.extend(metadata)
        else:
            self.document_metadata.extend([{} for _ in range(len(documents))])

        # Mark that we need to rebuild index
        self.index_built = False

    def build_index(self, vectors: np.ndarray = None):
        """Build the search index from vectors"""
        if vectors is not None:
            if self.vectors is None:
                self.vectors = vectors
            else:
                self.vectors = np.vstack([self.vectors, vectors])
        self.index_built = True

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Dot product divided by product of norms
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors"""
        return np.linalg.norm(vec1 - vec2)

    def search(self, query_vector: np.ndarray, top_k: int = 5,
               metric: str = 'cosine') -> List[Tuple[int, float, Dict]]:
        """Search for similar vectors"""
        if not self.index_built or self.vectors is None:
            raise ValueError("Index not built. Please build index first.")

        similarities = []

        for i, vec in enumerate(self.vectors):
            if metric == 'cosine':
                similarity = self.cosine_similarity(query_vector, vec)
            elif metric == 'euclidean':
                # Convert distance to similarity (inverse)
                distance = self.euclidean_distance(query_vector, vec)
                similarity = 1 / (1 + distance)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            similarities.append((i, similarity, self.document_metadata[i]))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_document(self, index: int) -> str:
        """Get document by index"""
        return self.documents[index]

    def save_index(self, path: str):
        """Save the index to disk"""
        data = {
            'documents': self.documents,
            'vectors': self.vectors.tolist() if self.vectors is not None else None,
            'metadata': self.document_metadata,
            'dimension': self.dimension
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    def load_index(self, path: str):
        """Load the index from disk"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.documents = data['documents']
        self.document_metadata = data['metadata']
        self.dimension = data['dimension']

        if data['vectors']:
            self.vectors = np.array(data['vectors'])
            self.index_built = True