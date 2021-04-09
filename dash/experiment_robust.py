
from dash.dash_ts import IDACS_TS
from dash.experiment_evaluate import *
from dash.experiment_evaluate import _classification

from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict

from pyts.transformation import ShapeletTransform


def run_grabocka(X_train, n_classes):

    # Set the number of shapelets per size as done in the original paper
    n_ts, ts_sz = X_train.shape
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz, n_classes=n_classes, l=0.1, r=1)

    print(datetime.datetime.now(), 'Training model')
    method = ShapeletModel(n_shapelets_per_size=shapelet_sizes, optimizer='sgd',
                           weight_regularizer=.01, max_iter=1000, verbose=0)
    window_sizes = list(shapelet_sizes.keys())
    return method, None, window_sizes


def run_learning(n_shapelets, window_sizes, window_steps):

    method = ShapeletTransform(n_shapelets=n_shapelets, window_sizes=window_sizes,
                               window_steps=window_steps, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_random(n_shapelets, window_sizes, window_steps):
    method = IDACS_TS(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering=None, train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash(n_shapelets, window_sizes, window_steps):
    method = IDACS_TS(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmeans', train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_experiment(D, dataset_name, method_name, n_trials=10):
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    n_classes = D['n_classes']
    window_sizes = D['window_sizes']
    window_steps = D['window_steps']

    if method_name == 'grabocka':
        method, _, ws = run_grabocka(X_train, n_classes)

    elif method_name == 'learning':
        method, n_sh, ws = run_learning(n_shapelets, window_sizes, window_steps)

    elif method_name == 'dash':
        method, n_sh, ws = run_dash(n_shapelets, window_sizes, window_steps)

    elif method_name == 'random':
        method, n_sh, ws = run_random(n_shapelets, window_sizes, window_steps)

    else:
        raise ValueError('Unknown method %s' % method_name)

    print(datetime.datetime.now(), 'Evaluating robusteness')
    time_train_list = list()
    shapelets_list = list()

    for i in range(n_trials):
        print(datetime.datetime.now(), i, n_trials)
        ts = time.time()
        method.fit(X_train, y_train)
        shapelets_list.append(method.shapelets_)

        time_train = time.time() - ts
        time_train_list.append(time_train)

    distances = list()

    for i in range(n_trials):
        shapelets_i = shapelets_list[i]

        for j in range(i+1, n_trials):
            shapelets_j = shapelets_list[j]

            for ii in range(len(shapelets_i)):
                shapelet_i = shapelets_i[ii]
                distance_i = np.inf

                for jj in range(len(shapelets_j)):
                    shapelet_j = shapelets_j[jj]

                    if len(shapelet_i) >= len(shapelet_j):
                        sl, sc = shapelet_i, shapelet_j
                    else:
                        sl, sc = shapelet_j, shapelet_i

                    # print('A', sl)
                    # print('B', sc)
                    steps = len(sl) - len(sc) + 1
                    # print(ii, jj, steps)
                    d = np.inf
                    for k in range(steps):
                        # print(k, 'C', sl[k:len(sc)+k])
                        d0 = np.sqrt(np.mean((sl[k:len(sc)+k] - sc) ** 2))
                        d = min(d, d0)
                    # print('\t', ii, jj, d, distances_i[ii], min(distances_i[ii], d))
                    distance_i = min(distance_i, d)
                # print(distances_i)
                distances.append(distance_i)

    distances = np.array(distances)

    print(datetime.datetime.now(), 'Model trained in %.2f' % np.mean(time_train_list))

    eval_clf = dict()
    eval_clf['dataset'] = dataset_name
    eval_clf['method'] = method_name
    eval_clf['n_shapelets'] = n_shapelets
    eval_clf['window_sizes'] = ws
    eval_clf['avg_time_train'] = np.mean(time_train_list)
    eval_clf['mean_dist'] = np.mean(distances)
    eval_clf['median_dist'] = np.median(distances)
    eval_clf['min_dist'] = np.min(distances)
    eval_clf['max_dist'] = np.max(distances)
    eval_clf['std_dist'] = np.std(distances)

    print(datetime.datetime.now(), 'Storing evaluation')
    store_result([eval_clf], 'robust_' + method_name)
    return 0


def main():
    method = 'learning'

    for dataset in ['gunpoint', 'italypower', 'ecg200', 'phalanges']:
        print(datetime.datetime.now(), 'Dataset: %s' % dataset)

        D = get_dataset(dataset, path_dataset, normalize)

        run_experiment(D, dataset, method)
        print('')


if __name__ == "__main__":
    main()
