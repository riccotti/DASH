
import numpy as np
from itertools import chain
from collections import defaultdict

from numba import njit, prange

from itertools import combinations

from joblib import delayed, Parallel

from sklearn.feature_selection import mutual_info_classif

from dash.dash_base import IDACS

@njit
def _derive_base_distances_fit(X, prototype):
    n_samples, n_features = X.shape
    base_distances = np.zeros((n_samples, n_features))
    for i in prange(n_features):
        for j in prange(n_samples):
            base_distances[j, i] = (X[j, i] - prototype[i]) ** 2
    return base_distances

@njit
def _derive_shapelet_one_combination_fit(base_distances_fs, prototypes_fs, n_prototypes, features_sel):
    distances = []
    shapelets = []
    features_sel_list = []
    prototype_idx = []
    n_samples = base_distances_fs.shape[1]
    # print('comb', features_sel)
    for p_idx in prange(n_prototypes):
        distances_in = []
        for s_idx in prange(n_samples):
            dist = np.sqrt(np.mean(base_distances_fs[p_idx, s_idx]))
            distances_in.append(dist)
            # if s_idx <= 10:
            #     print(p_idx, s_idx, dist, base_distances_fs[p_idx, s_idx])
        distances.append(distances_in)
        features_sel_list.append(features_sel)
        shapelets.append(prototypes_fs[p_idx])
        prototype_idx.append(p_idx)

    return distances, shapelets, features_sel_list, prototype_idx

@njit
def _derive_distance_transform_shapelet(X, shapelet, n_samples):
    distances = np.zeros(n_samples)
    for j in prange(n_samples):
        distances[j] = np.sqrt(np.mean((X[j] - shapelet) ** 2))
    return distances


def _derive_distances_transform(X, shapelets, indices, n_samples, n_shapelets):
    distances = np.zeros((n_shapelets, n_samples))
    for i in prange(n_shapelets):
        distances[i] = _derive_distance_transform_shapelet(X[:, indices[i]], shapelets[i], n_samples)
    return distances


# @njit
# def _isin(A, B, n, m):
#     for i in prange(m):
#         if i + n >= m:
#             break
#         for j in prange(n):
#             if A[j] != B[i + j]:
#                 break
#             if j == n - 1:
#                 return 1
#     return 0
#
#
# @njit
# def _check_apriori(features_combinations, n_combinations,
#                    top_n_features_keys, top_n_features_values, n_top_n_features):
#     features_combinations_score = np.zeros(n_combinations)
#
#     for i in prange(n_combinations):
#         for j in prange(n_top_n_features):
#             if _isin(top_n_features_keys[j], features_combinations[i],
#                      len(top_n_features_keys[j]), len(features_combinations[i])) == 1:
#                 features_combinations_score[i] = features_combinations_score[i] + top_n_features_values[j]
#
#     return features_combinations_score


class IDACS_TAB(IDACS):

    def __init__(self, window_sizes, n_shapelets=1, n_clusters=2, max_comb=10000, apriori_like=True, top_n=1000,
                 clustering='kmeans', train_set='sample', random_state=None, n_jobs=-1, verbose=None):
        self.max_comb = max_comb
        self.apriori_like = apriori_like
        self.top_n = top_n
        super().__init__(window_sizes, n_shapelets, n_clusters, clustering, train_set, random_state, n_jobs, verbose)

    def _fit(self, X, y, window_sizes=None):

        window_sizes = self.window_sizes if window_sizes is None else window_sizes

        scores, shapelets, indices = self._extract_shapelets(X, y, window_sizes)

        self.scores_ = scores
        self.shapelets_ = shapelets
        self.indices_ = indices

        return self

    # def _check_apriori(self, top_n_features, features_combinations, f0):
    #     features_combinations_score = defaultdict(float)
    #     for f1 in features_combinations:
    #         if set(f0) <= set(f1):
    #             features_combinations_score[f1] += top_n_features[f0]
    #     return features_combinations_score

    def _extract_shapelets(self, X, y, window_sizes):
        prototypes = self.prototypes
        n_prototypes = len(prototypes)

        n_samples, n_features = X.shape
        base_distances_list = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._derive_base_distances_one_prototype)(X, p_idx)
            for p_idx in range(n_prototypes))
        # base_distances_list = list()
        # for p_idx in range(n_prototypes):
        #     base_distances_list.append(self._derive_base_distances_fit_in(X, p_idx))

        base_distances = np.stack(base_distances_list)
        # print(len(base_distances))

        res_all = list()
        top_n_features = dict()

        # from scipy.special import comb

        for wi, window_size in enumerate(window_sizes):

            features_combinations = [tuple([f]) for f in np.arange(n_features)]

            if self.apriori_like:
                if wi > 0:
                    features_combinations_score = defaultdict(float)
                    for if0, f0 in enumerate(top_n_features.keys()):
                        for if1, f1 in enumerate(top_n_features.keys()):
                            if if1 < if0:
                                continue
                            f_new = tuple(set([f for f in f0] + [f for f in f1]))
                            if len(f_new) == window_size:
                                features_combinations_score[f_new] += top_n_features[f0] + top_n_features[f1]

                    idx = set(np.argsort(list(features_combinations_score.values()))[-self.top_n:])

                    features_combinations_new = list()
                    for i, k in enumerate(features_combinations_score.keys()):
                        if i in idx:
                            features_combinations_new.append(k)
                    features_combinations = features_combinations_new
            # print(features_combinations)

            # versione vecchai ok
            # features_combinations = list(combinations(np.arange(n_features), window_size))
            #
            # if self.apriori_like and len(top_n_features) > 0:
            #     # # print('apriori')
            #     features_combinations_score = defaultdict(float)
            #     for f1 in features_combinations:
            #         for f0 in top_n_features:
            #             if set(f0) <= set(f1):
            #                 features_combinations_score[f1] += top_n_features[f0]
            #
            #     idx = set(np.argsort(list(features_combinations_score.values()))[-self.top_n:])
            #
            #     features_combinations_new = list()
            #     for i, k in enumerate(features_combinations_score.keys()):
            #         if i in idx:
            #             features_combinations_new.append(k)
            #     features_combinations = features_combinations_new
            # print(features_combinations)

            if len(features_combinations) > self.max_comb:
                # print('upper bound')
                features_combinations_idx = np.random.choice(len(features_combinations),
                                                             self.max_comb, replace=False)
                features_combinations_new = list()
                for f in features_combinations_idx:
                    features_combinations_new.append(features_combinations[f])

                features_combinations = features_combinations_new
            # print('after', len(features_combinations))

            res = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._derive_shapelet_one_combination)(
                    base_distances, y, prototypes, n_prototypes, features_sel)
                for features_sel in features_combinations)

            (shapelets, features_sel_list, scores, prototype_idx) = zip(*res)

            shapelets = np.asarray([list(shapelet) for shapelet in shapelets])
            shapelets = np.asarray(list(chain.from_iterable(shapelets)))
            features_sel_list = np.asarray([features_sel for features_sel in features_sel_list])
            features_sel_list = np.asarray(list(chain.from_iterable(features_sel_list)))
            scores = np.concatenate(scores)
            prototype_idx = np.concatenate(prototype_idx)

            if self.apriori_like:
                # idx = np.argpartition(scores, scores.size - self.top_n)[-self.top_n:]
                for f, s in zip(features_sel_list, scores):
                    # print(f, s)
                    top_n_features[tuple(f)] = max(s, top_n_features.get(tuple(f), -1.0))

                idx = set(np.argsort(list(top_n_features.values()))[-self.top_n:])
                # print(idx)
                top_n_features_new = dict()
                for i, k in enumerate(top_n_features.keys()):
                    if i in idx:
                        top_n_features_new[k] = top_n_features[k]
                top_n_features = top_n_features_new

            # print(shapelets.shape)
            # print(features_sel_list.shape)
            # print(scores.shape)
            # print(prototype_idx.shape)
            #
            # print(shapelets[0])
            # print(features_sel_list[0])
            # print(scores[0])
            # print(prototype_idx[0])

            if scores.size > self.n_shapelets - 1:
                idx = np.argpartition(scores, scores.size - self.n_shapelets)[-self.n_shapelets:]
                scores = scores[idx]
                shapelets = shapelets[idx]
                features_sel_list = features_sel_list[idx]
                prototype_idx = prototype_idx[idx]

            res_all.append((shapelets, features_sel_list, scores, prototype_idx))

        (shapelets, features_sel_list, scores, prototype_idx) = zip(*res_all)

        shapelets = np.asarray([list(shapelet) for shapelet in shapelets])
        shapelets = np.asarray(list(chain.from_iterable(shapelets)))
        features_sel_list = np.asarray([list(features_sel) for features_sel in features_sel_list])
        features_sel_list = np.asarray(list(chain.from_iterable(features_sel_list)))
        scores = np.concatenate(scores)
        prototype_idx = np.concatenate(prototype_idx)

        if scores.size > self.n_shapelets - 1:
            idx = np.argpartition(scores, scores.size - self.n_shapelets)[-self.n_shapelets:]
            scores = scores[idx]
            shapelets = shapelets[idx]
            features_sel_list = features_sel_list[idx]
            prototype_idx = prototype_idx[idx]

        # Derive the 'indices' attributes
        # print(features_sel_list, type(features_sel_list))
        # indices = np.empty((scores.size, 2), dtype='int64')
        # indices[:, 0] = prototype_idx
        # indices[:, 1] = list(features_sel_list)
        indices = features_sel_list

        return scores, shapelets, indices

    def _derive_base_distances_one_prototype(self, X, p_idx):
        prototype = self.prototypes[p_idx]
        bd = _derive_base_distances_fit(X, tuple(prototype))
        return bd

    def _derive_shapelet_one_combination(self, base_distances, y, prototypes, n_prototypes, features_sel):

        base_distances_fs = base_distances[:, :, features_sel]
        prototypes_fs = prototypes[:, features_sel]

        res = _derive_shapelet_one_combination_fit(
            base_distances_fs, prototypes_fs, n_prototypes, features_sel)

        distances, shapelets, features_sel_list, prototype_idx = res

        X_dist = np.asarray(distances).T
        scores = mutual_info_classif(X_dist, y)
        return shapelets, features_sel_list, scores, prototype_idx

    def _transform(self, X):
        n_samples = X.shape[0]
        n_shapelets = self.n_shapelets
        shapelets = self.shapelets_
        indices = self.indices_

        X_dist = _derive_distances_transform(X, tuple(shapelets), tuple(indices), n_samples, n_shapelets)
        return X_dist.T

    def _locate(self, X):
        locs = [self.indices_] * len(X)
        return locs
