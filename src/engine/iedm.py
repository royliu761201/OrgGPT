from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class IEDM:
    """
    Entropy calculation layer for OrgGPT.
    Uses TF-IDF term weights for message distributions, Jensen-Shannon
    divergence for semantic drift, and Shannon entropy over K-Means clusters for
    fragmentation.
    """

    def __init__(self, max_features: int = 512, smoothing: float = 1e-12):
        if max_features < 1:
            raise ValueError("max_features must be at least 1.")
        if smoothing < 0.0:
            raise ValueError("smoothing must be non-negative.")

        self.max_features = max_features
        self.smoothing = smoothing

    def get_embedding(self, text: str) -> str:
        # TF-IDF is fit over the local corpus, so the raw text is the embedding payload.
        return text

    def _build_vectorizer(self) -> TfidfVectorizer:
        # A fresh vectorizer per fit keeps the class free of shared mutable state.
        return TfidfVectorizer(max_features=self.max_features, norm=None, dtype=np.float64)

    def _fit_transform(self, corpus: list[str]) -> csr_matrix:
        if not corpus:
            return csr_matrix((0, 0), dtype=np.float64)

        try:
            return self._build_vectorizer().fit_transform(corpus).tocsr()
        except ValueError as exc:
            if "empty vocabulary" in str(exc):
                return csr_matrix((len(corpus), 0), dtype=np.float64)
            raise

    def _to_probability_matrix(self, matrix: csr_matrix) -> np.ndarray:
        if matrix.shape[0] == 0:
            return np.empty((0, 0), dtype=np.float64)
        if matrix.shape[1] == 0:
            return np.ones((matrix.shape[0], 1), dtype=np.float64)

        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        normalized = normalize(matrix, norm="l1", axis=1, copy=True)
        dense = normalized.toarray().astype(np.float64, copy=False)

        zero_rows = row_sums <= self.smoothing
        if np.any(zero_rows):
            dense[zero_rows] = 1.0 / matrix.shape[1]

        return dense

    def _has_variation(self, matrix: csr_matrix) -> bool:
        if matrix.shape[0] <= 1:
            return False

        reference = matrix.getrow(0)
        for row_index in range(1, matrix.shape[0]):
            if (matrix.getrow(row_index) != reference).nnz > 0:
                return True
        return False

    def compute_distortion(self, v_root: str, V_level: list[str]) -> float:
        """
        d_l = E_i[ JS(P_0 || P_i) ] where P_0 and P_i are TF-IDF term
        distributions. scipy.spatial.distance.jensenshannon returns the metric
        distance sqrt(JS), so we square it to recover the divergence in bits.
        """
        if not V_level:
            return 0.0

        matrix = self._fit_transform([v_root] + V_level)
        if matrix.shape[1] == 0:
            return 0.0

        distributions = self._to_probability_matrix(matrix)
        root_distribution = np.repeat(distributions[[0]], repeats=len(V_level), axis=0)
        child_distributions = distributions[1:]

        js_distance = jensenshannon(
            root_distribution,
            child_distributions,
            axis=1,
            base=2.0,
        )
        js_divergence = np.square(js_distance)
        return float(np.mean(js_divergence))

    def compute_entropy(self, V_level: list[str], k_clusters: int = 3) -> float:
        """
        H_l = -sum(p_i * log2(p_i)) over K-Means clusters at hierarchy depth l.
        """
        if len(V_level) <= 1:
            return 0.0

        matrix = self._fit_transform(V_level)
        if matrix.shape[1] == 0 or not self._has_variation(matrix):
            return 0.0

        n_clusters = min(max(1, k_clusters), len(V_level))
        if n_clusters == 1:
            return 0.0

        normalized = normalize(matrix, norm="l2", axis=1, copy=False)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized)

        counts = np.bincount(labels, minlength=n_clusters)
        probs = counts[counts > 0] / len(labels)
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)
