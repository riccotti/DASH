
import numpy as np
from abc import ABC, abstractmethod

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class IDACS(ABC):

    def __init__(self, window_sizes, n_shapelets=1, n_clusters=2,
                 clustering='kmeans', train_set='sample', random_state=None, n_jobs=-1, verbose=None):
        self.window_sizes = tuple(window_sizes) if not isinstance(window_sizes, str) else window_sizes
        self.n_shapelets = n_shapelets
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.train_set = train_set
        self.scaler = StandardScaler()

        if clustering is None or clustering in ['rnd', 'random']:
            self.clustering = None
        elif clustering in ['kmeans', 'k-means']:
            self.clustering = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10,
                                     max_iter=250, tol=1e-4, n_jobs=self.n_jobs, random_state=self.random_state)
        elif clustering in ['kmedoids', 'k-medoids']:
            self.clustering = KMedoids(n_clusters=n_clusters, metric='euclidean', init='heuristic',
                                       max_iter=250, random_state=self.random_state)
        else:
            raise ValueError('Unknown clusteirng %s' % clustering)

        super().__init__()

    @abstractmethod
    def _fit(self, X, y):
        pass

    @abstractmethod
    def _transform(self, X):
        pass

    @abstractmethod
    def _locate(self, X):
        pass

    def _sampling(self, X, y, sample_size):
        if self.clustering is not None and sample_size is not None and sample_size < len(X):
            X_s, _, y_s, _ = train_test_split(X, y, train_size=sample_size,
                                              random_state=self.random_state,
                                              stratify=y)
        elif self.clustering is None:
            X_s, _, y_s, _ = train_test_split(X, y, train_size=0.1,
                                              random_state=self.random_state,
                                              stratify=y)
        else:
            X_s, y_s = X, y

        return X_s, y_s

    def _prototype_extraction(self, X, y):
        if self.clustering is None:
            prototypes = list()
            for prototype in X:
                prototypes.append(prototype)
        else:
            if X.ndim > 2:
                s0, s1, s2 = X.shape
                X_r = X.reshape(s0, s1 * s2)
            else:
                X_r = X

            X_s = self.scaler.fit_transform(X_r)
            class_values = sorted(np.unique(y))

            prototypes = list()
            for label in class_values:
                self.clustering.fit(X_s[np.where(y == label)[0]])
                cluster_centers = self.scaler.inverse_transform(self.clustering.cluster_centers_)
                for prototype in cluster_centers:
                    if X.ndim > 2:
                        prototype = prototype.reshape((s1, s2))
                    prototypes.append(prototype)

        return np.array(prototypes)

    def fit(self, X, y, sample_size=10000):

        X_s, y_s = self._sampling(X, y, sample_size)

        self.prototypes = self._prototype_extraction(X_s, y_s)

        if self.train_set == 'sample':
            X_u, y_u = X_s, y_s
        elif self.train_set == 'all':
            X_u, y_u = X, y
        else:
            raise ValueError('Unknown train set %s' % self.train_set)

        self._fit(X_u, y_u)

        return self

    def transform(self, X):
        return self._transform(X)

    def locate(self, X):
        return self._locate(X)


