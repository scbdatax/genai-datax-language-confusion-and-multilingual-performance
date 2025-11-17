import numpy as np
from typing import Tuple


class EigenUncertainty:
    @staticmethod
    def get_L_mat(W: np.array) -> np.array:
        """
        Calculate the normalized laplacian matrix (L) from the degree matrix (D) and weighted adjacency matrix (W).

        Args:
            W (np.array): weighted adjacency matrix.

        Returns
            normalized laplacian matrix (L).
        """
        # compute the degreee matrix from the weighted adjacency matrix
        D = np.diag(np.sum(W, axis=1))
        D_inv = np.linalg.inv(np.sqrt(D))
        L = D_inv @ (D - W) @ D_inv
        # L = np.identity(n=D.shape[0]) - D_inv @ W @ D_inv
        return L.copy()

    @staticmethod
    def get_eig(
        L: np.array, 
        thres: float = 1., 
        eps: float = 1e-4,
    ) -> Tuple[np.array]:
        """
        Calculate the eigenvalues and eigenvectors of the normalized laplacian matrix (L).

        Args:
            L (np.array): normalized laplacian matrix (assuming it's symmetric).
            thres (float): threshold to cut off those eigenvalues which are pretty high values.
            eps (float): small offset to avoid singularity.

        Returns:
            tuple of eigenvalues and eigenvectors arrays.
        """
        if eps is not None:
            L = (1-eps) * L + eps * np.eye(len(L))
        eigvals, eigvecs = np.linalg.eigh(L)

        if thres is not None:
            keep_mask = eigvals < thres
            eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
        return eigvals, eigvecs

    def estimate(self, W: np.array) -> float:
        """
        Estimate uncertainty from top k eigenvalues from the normalized laplacian matrix.

        Args:
            all_responses (list): list of N original respones.

        Returns:
            uncertainty estimation using top k eigenvalues.
        """
        L = self.get_L_mat(W)
        eigvals, _ = self.get_eig(L)
        return np.sum(np.maximum(0, 1 - eigvals))
