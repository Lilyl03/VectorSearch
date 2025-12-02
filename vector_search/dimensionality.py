# vector_search/dimensionality.py
import numpy as np
from typing import Tuple


class DimensionalityReducer:
    """Implement PCA for dimensionality reduction"""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X: np.ndarray) -> 'DimensionalityReducer':
        """Fit PCA to the data"""
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Store principal components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimension"""
        if self.mean is None or self.components is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio"""
        if self.explained_variance is None:
            return None
        return self.explained_variance / np.sum(self.explained_variance)


class SVDReducer:
    """Implement SVD for dimensionality reduction"""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.VT = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Perform SVD and return reduced dimensions"""
        # Center the data
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        # Perform SVD
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

        # Store components
        self.U = U[:, :self.n_components]
        self.S = S[:self.n_components]
        self.VT = VT[:self.n_components, :]

        # Return reduced representation
        return self.U * self.S