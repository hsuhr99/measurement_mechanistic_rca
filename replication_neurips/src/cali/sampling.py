import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    from sklearn.neighbors import NearestNeighbors

from .utils import markov_blanket

class FeatureSampler:
    def __init__(self, data, dag):
        self.data = data.astype(np.float32)  # FAISS needs float32
        self.dag = dag
        self.n_samples, self.n_features = data.shape

    def sample(self, feature_idx, target_idx=None, k=5):
        raise NotImplementedError("Must implement sample method")


class RandomSampler(FeatureSampler):
    def sample(self, feature_ids, target_idx=None, k=5):
        indices = np.random.choice(self.n_samples, size=k, replace=False)
        return self.data[indices, feature_ids]


class KNNSampler(FeatureSampler):
    def __init__(self, data, dag, parent_child_map=None):
        super().__init__(data, dag)
        self.markov_blanket_map = self._compute_mb_map()
        self.feature_queries = {}
        self.old_queries = {}

    def _compute_mb_map(self):
        map_ = {}
        for i in range(self.n_features):
            map_[i] = markov_blanket(self.dag, i)
        return map_

    def _get_faiss_index(self, queried_features):
        if queried_features not in self.feature_queries:
            query_data = self.data[:, queried_features].astype(np.float32)
            if faiss is None:
                index = NearestNeighbors(metric="euclidean")
                index.fit(query_data)
            else:
                d = len(queried_features)
                index = faiss.IndexFlatL2(d)  # exact KNN, L2 distance
                index.add(query_data)
            self.feature_queries[queried_features] = index
        return self.feature_queries[queried_features]

    def sample(self, feature_ids, target_idx=None, k=5):
        if target_idx is None:
            raise ValueError("No target_idx provided")

        query_tuple = (target_idx, tuple(sorted(feature_ids)))
        if query_tuple not in self.old_queries:
            queried_features = set()
            for feature_idx in feature_ids:
                queried_features.update(self.markov_blanket_map[feature_idx])
            for feature_idx in feature_ids:
                queried_features.discard(feature_idx)
            queried_features = tuple(sorted(queried_features))

            if not queried_features:
                raise ValueError("No valid queried_features from markov blanket")

            index = self._get_faiss_index(queried_features)
            query_vec = self.data[target_idx, queried_features].astype(np.float32).reshape(1, -1)

            if faiss is None:
                _, neighbors_idx = index.kneighbors(query_vec, n_neighbors=k)
            else:
                _, neighbors_idx = index.search(query_vec, k)
            self.old_queries[query_tuple] = neighbors_idx[0]

        return self.old_queries[query_tuple]
