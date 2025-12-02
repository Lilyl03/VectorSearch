# vector_search/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from mpl_toolkits.mplot3d import Axes3D


class SearchVisualizer:
    """Visualize vector search results and dimensionality reduction"""

    @staticmethod
    def plot_vectors_2d(vectors: np.ndarray, labels: Optional[List[str]] = None,
                        title: str = "2D Vector Visualization"):
        """Plot vectors in 2D space"""
        plt.figure(figsize=(10, 8))

        if vectors.shape[1] != 2:
            raise ValueError("Vectors must be 2D for this visualization")

        plt.scatter(vectors[:, 0], vectors[:, 1], alpha=0.6, s=100)

        if labels:
            for i, label in enumerate(labels):
                plt.annotate(label, (vectors[i, 0], vectors[i, 1]),
                             fontsize=8, alpha=0.7)

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_vectors_3d(vectors: np.ndarray, labels: Optional[List[str]] = None,
                        title: str = "3D Vector Visualization"):
        """Plot vectors in 3D space"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if vectors.shape[1] != 3:
            raise ValueError("Vectors must be 3D for this visualization")

        ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2],
                   alpha=0.6, s=100)

        if labels:
            for i, label in enumerate(labels):
                ax.text(vectors[i, 0], vectors[i, 1], vectors[i, 2],
                        label, fontsize=8, alpha=0.7)

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title(title)
        plt.show()

    @staticmethod
    def plot_search_results(query_vector: np.ndarray, results: np.ndarray,
                            original_dims: int = 300,
                            title: str = "Search Results Visualization"):
        """Visualize query and search results"""
        # Reduce dimensions for visualization
        from sklearn.decomposition import PCA

        # Combine query and results
        all_vectors = np.vstack([query_vector.reshape(1, -1), results])

        # Reduce to 2D
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(all_vectors)

        plt.figure(figsize=(10, 8))

        # Plot query vector
        plt.scatter(reduced_vectors[0, 0], reduced_vectors[0, 1],
                    color='red', s=200, marker='*', label='Query', edgecolors='black')

        # Plot results
        plt.scatter(reduced_vectors[1:, 0], reduced_vectors[1:, 1],
                    color='blue', s=100, alpha=0.6, label='Results')

        # Draw lines from query to results
        for i in range(1, len(reduced_vectors)):
            plt.plot([reduced_vectors[0, 0], reduced_vectors[i, 0]],
                     [reduced_vectors[0, 1], reduced_vectors[i, 1]],
                     'gray', alpha=0.3, linestyle='--')

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()