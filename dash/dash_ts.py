
import numpy as np
from itertools import chain

from numba import njit, prange
from numpy.lib.stride_tricks import as_strided

from joblib import delayed, Parallel

from sklearn.feature_selection import mutual_info_classif

from dash.dash_base import IDACS


@njit()
def _extract_all_shapelets_raw(x, window_sizes, window_steps, n_timestamps):
    """Extract all the shapelets from alphabet_size single time series prototype."""
    shapelets = []  # shapelets
    lengths = []  # lengths of shapelets
    start_idx = []  # start index of shapelets (included)
    end_idx = []  # end index of shapelets (excluded)

    for window_size, window_step in zip(window_sizes, window_steps):
        # Derive the new shape and strides
        overlap = window_size - window_step
        shape_new = ((n_timestamps - overlap) // window_step, window_size // 1)
        strides = x.strides[0]
        strides_new = (window_step * strides, strides)

        # Derive strided view of x
        x_strided = as_strided(x, shape=shape_new, strides=strides_new)
        x_strided = np.copy(x_strided)

        # Add shapelets, lengths, start indices and end indices
        shapelets.append(x_strided)
        lengths.append([x_strided.shape[1]] * x_strided.shape[0])
        start_idx.append(np.arange(0, n_timestamps - window_size + 1,
                                   window_step))
        end_idx.append(np.arange(window_size, n_timestamps + 1, window_step))

    return shapelets, lengths, start_idx, end_idx


def _extract_all_shapelets(x, window_sizes, window_steps, n_timestamps):
    """Extract all the shapelets from alphabet_size single time series prototype."""
    shapelets, lengths, start_idx, end_idx = _extract_all_shapelets_raw(
        x, window_sizes, window_steps, n_timestamps)

    # Convert list to tuple
    shapelets = tuple(shapelets)
    lengths = tuple(lengths)

    return shapelets, lengths, start_idx, end_idx


@njit()
def _windowed_view(X, n_samples, n_timestamps, window_size, window_step):
    overlap = window_size - window_step
    shape_new = (n_samples,
                 (n_timestamps - overlap) // window_step,
                 window_size // 1)
    s0, s1 = X.strides
    strides_new = (s0, window_step * s1, s1)
    return as_strided(X, shape=shape_new, strides=strides_new)


@njit()
def _derive_shapelet_distances(X, shapelet):
    """Derive the distance between alphabet_size shapelet and all the time series."""
    n_samples, n_windows, _ = X.shape
    mean = np.empty((n_samples, n_windows))
    for i in prange(n_samples):
        for j in prange(n_windows):
            mean[i, j] = np.mean((X[i, j] - shapelet) ** 2)
    dist = np.empty(n_samples)
    for i in prange(n_samples):
        dist[i] = np.min(mean[i])
    return dist


@njit()
def _derive_shapelet_distances_locate(X, shapelet):
    """Derive the distance between alphabet_size shapelet and all the time series."""
    n_samples, n_windows, _ = X.shape
    mean = np.empty((n_samples, n_windows))
    for i in prange(n_samples):
        for j in prange(n_windows):
            mean[i, j] = np.mean((X[i, j] - shapelet) ** 2)
    location = np.empty(n_samples)
    for i in prange(n_samples):
        location[i] = np.argmin(mean[i])
    return location


@njit()
def _derive_all_squared_distances_fit(
    X, n_samples, n_timestamps, shapelets, lengths, window_step=1):
    """Derive the squared distances between all shapelets and time series."""
    distances = []  # save the distances in alphabet_size list
    # distances = List()

    for i in prange(len(lengths)):
        window_size = lengths[i][0]
        X_window = _windowed_view(X, n_samples, n_timestamps, window_size, window_step)
        for j in prange(shapelets[i].shape[0]):
            dist = _derive_shapelet_distances(X_window, shapelets[i][j])
            distances.append(dist)

    return distances


def _derive_all_distances_fit(
    X, n_samples, n_timestamps, shapelets, lengths, window_step=1):
    """Derive the distances between all the shapelets and the time series."""
    distances = _derive_all_squared_distances_fit(
        X, n_samples, n_timestamps, shapelets, lengths, window_step)
    return np.sqrt(np.asarray(distances)).T


@njit()
def _derive_all_squared_distances_transform(
    X, n_samples, n_timestamps, window_sizes, shapelets, lengths, window_step=1):
    """Derive the squared distances between all shapelets and time series."""
    distances = []  # save the distances in alphabet_size list
    permutation = []  # save the permutation of the indices

    for window_size in window_sizes:
        X_window = _windowed_view(X, n_samples, n_timestamps, window_size, window_step)
        indices = np.where(lengths == window_size)[0]
        permutation.append(indices)

        for idx in indices:
            dist = _derive_shapelet_distances(X_window, shapelets[idx])
            distances.append(dist)

    return distances, permutation


@njit()
def _derive_all_squared_distances_locate(
        X, n_samples, n_timestamps, window_sizes, shapelets, lengths, window_step=1):
    """Derive the squared distances between all shapelets and time series."""
    locations = []
    permutation = []  # save the permutation of the indices

    for window_size in window_sizes:
        X_window = _windowed_view(X, n_samples, n_timestamps, window_size, window_step)
        indices = np.where(lengths == window_size)[0]
        permutation.append(indices)

        for idx in indices:
            locs = _derive_shapelet_distances_locate(X_window, shapelets[idx])
            locations.append(locs)

    return locations, permutation


class IDACS_TS(IDACS):

    def __init__(self, window_sizes, window_steps=None, n_shapelets=1, n_clusters=2,
                 clustering='kmeans', train_set='all', random_state=None, n_jobs=-1, verbose=None):
        self.window_steps = tuple(window_steps) if window_steps is not None else window_steps
        super().__init__(window_sizes, n_shapelets, n_clusters, clustering, train_set, random_state, n_jobs, verbose)

    def _fit(self, X, y, window_sizes=None, window_steps=None):

        window_sizes = self.window_sizes if window_sizes is None else window_sizes
        window_steps = self.window_steps if window_steps is None else window_steps

        if isinstance(window_sizes, str):
            if window_sizes == 'auto':
                window_sizes, window_steps = self._auto_length_computation(X, y)
                self.window_sizes = window_sizes
                self.window_steps = window_steps

        scores, shapelets, indices = self._extract_shapelets(X, y, window_sizes, window_steps)

        self.scores_ = scores
        self.shapelets_ = shapelets
        self.indices_ = indices

        return self

    def _extract_shapelets(self, X, y, window_sizes, window_steps):
        """Fit all the time series"""
        n_prototypes = len(self.prototypes)
        n_samples, n_timestamps = X.shape
        res = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._extract_shapelet_one_prototype)(X, y, p_idx,
                                                          n_samples, n_timestamps,
                                                          window_sizes, window_steps)
            for p_idx in range(n_prototypes))

        (X_dist, scores, shapelets,
         start_idx, end_idx, prototype_idx) = zip(*res)

        # Concatenate the results
        X_dist = np.hstack(X_dist)
        scores = np.concatenate(scores)
        shapelets = np.concatenate(shapelets)
        start_idx = np.concatenate(start_idx)
        end_idx = np.concatenate(end_idx)
        prototype_idx = np.concatenate(prototype_idx)

        # Keep at most 'n_shapelets'
        if scores.size > self.n_shapelets - 1:
            idx = np.argpartition(scores, scores.size - self.n_shapelets)[-self.n_shapelets:]
            X_dist = X_dist[:, idx]
            scores = scores[idx]
            shapelets = shapelets[idx]
            start_idx = start_idx[idx]
            end_idx = end_idx[idx]
            prototype_idx = prototype_idx[idx]

        # Derive the 'indices' attributes
        indices = np.empty((scores.size, 3), dtype='int64')
        indices[:, 0] = prototype_idx
        indices[:, 1] = start_idx
        indices[:, 2] = end_idx

        return scores, shapelets, indices

    def _extract_shapelet_one_prototype(self, X, y, p_idx, n_samples, n_timestamps, window_sizes, window_steps):
        """Fit one time series."""
        x = self.prototypes[p_idx]
        window_step_d = np.min(self.window_steps, axis=0)

        # Extract all shapelets
        shapelets, lengths, start_idx, end_idx = _extract_all_shapelets(
            x, window_sizes, window_steps, n_timestamps)

        # Derive distances between shapelets and time series
        X_dist = _derive_all_distances_fit(
            X, n_samples, n_timestamps, shapelets, lengths, window_step_d)

        # Calculate Mutual Information Classification
        scores = mutual_info_classif(X_dist, y, discrete_features=False, random_state=self.random_state)

        # Flatten the list of 2D arrays into an array of 1D arrays
        shapelets = [list(shapelet) for shapelet in shapelets]
        shapelets = np.asarray(list(chain.from_iterable(shapelets)))

        # Concatenate the list/tuple of 1D arrays into one 1D array
        start_idx = np.concatenate(start_idx)
        end_idx = np.concatenate(end_idx)

        # Keep at most 'n_shapelets'
        if scores.size > self.n_shapelets - 1:
            idx = np.argpartition(scores, scores.size - self.n_shapelets)[-self.n_shapelets:]
            scores = scores[idx]
            shapelets = shapelets[idx]
            start_idx = start_idx[idx]
            end_idx = end_idx[idx]
            X_dist = X_dist[:, idx]

        prototype_idx = np.full(scores.size, p_idx)
        return X_dist, scores, shapelets, start_idx, end_idx, prototype_idx

    def _transform(self, X):
        lengths = self.indices_[:, 2] - self.indices_[:, 1]
        window_sizes = np.unique(lengths)
        n_samples, n_timestamps = X.shape
        window_step_d = np.min(self.window_steps, axis=0)

        distances, permutation = _derive_all_squared_distances_transform(
            X, n_samples, n_timestamps, window_sizes, tuple(self.shapelets_), lengths, window_step_d)

        # Compute the inverse permutation of the indices
        permutation = np.concatenate(permutation)
        inverse_perm = np.arange(permutation.size)[np.argsort(permutation)]

        return np.sqrt(np.asarray(distances))[inverse_perm].T

    def _locate(self, X):
        lengths = self.indices_[:, 2] - self.indices_[:, 1]
        window_sizes = np.unique(lengths)
        n_samples, n_timestamps = X.shape
        window_step_d = np.min(self.window_steps, axis=0)

        locations, permutation = _derive_all_squared_distances_locate(
            X, n_samples, n_timestamps, window_sizes, tuple(self.shapelets_), lengths, window_step_d)

        permutation = np.concatenate(permutation)
        inverse_perm = np.arange(permutation.size)[np.argsort(permutation)]

        return np.asarray(locations)[inverse_perm].T

    def _auto_length_computation(self, X, y):
        """Derive the window sizes automatically."""
        n_samples, n_timestamps = X.shape
        window_sizes = np.arange(1, n_timestamps//2 + 1)
        window_steps = np.ones_like(window_sizes)

        shapelet_lengths = []
        for i in range(10):
            idx = np.random.choice(n_samples, size=10, replace=False)
            X_small, y_small = X[idx], y[idx]
            _, shapelets, _ = self._extract_shapelets(X_small, y_small, window_sizes, window_steps)
            shapelet_lengths.extend([len(shapelet) for shapelet in shapelets])

        window_range = np.percentile(
            shapelet_lengths, [25, 75], interpolation='lower'
        ).astype('int64')

        window_sizes = np.arange(window_range[0], window_range[1] + 1)
        window_steps = np.ones_like(window_sizes)

        return window_sizes, window_steps