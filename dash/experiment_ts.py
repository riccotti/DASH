
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


def run_random_rnd(n_shapelets, window_sizes, window_steps):
    method = IDACS_TS(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering=None, train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash(n_shapelets, window_sizes, window_steps):
    method = IDACS_TS(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmeans', train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash_med(n_shapelets, window_sizes, window_steps):
    method = IDACS_TS(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmedoids', train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash_rnd(n_shapelets, window_sizes, window_steps):
    method = IDACS_TS(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmeans', train_set='sample',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash_rnd_med(n_shapelets, window_sizes, window_steps):
    method = IDACS_TS(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmedoids', train_set='sample',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_experiment(D, dataset_name, method_name):
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

    elif method_name == 'dash_md':
        method, n_sh, ws = run_dash_med(n_shapelets, window_sizes, window_steps)

    elif method_name == 'dash_rnd':
        method, n_sh, ws = run_dash_rnd(n_shapelets, window_sizes, window_steps)

    elif method_name == 'dash_rnd_md':
        method, n_sh, ws = run_dash_rnd_med(n_shapelets, window_sizes, window_steps)

    elif method_name == 'random':
        method, n_sh, ws = run_random(n_shapelets, window_sizes, window_steps)

    elif method_name == 'random_rnd':
        method, n_sh, ws = run_random_rnd(n_shapelets, window_sizes, window_steps)

    else:
        raise ValueError('Unknown method %s' % method_name)

    ts = time.time()
    method.fit(X_train, y_train)
    if method_name == 'grabocka':
        n_sh = len(method.shapelets_)

    time_train = time.time() - ts
    print(datetime.datetime.now(), 'Model trained in %.2f' % time_train)

    print(datetime.datetime.now(), 'Transforming data')
    ts = time.time()
    X_train_d = method.transform(X_train)
    time_dtrain = time.time() - ts

    ts = time.time()
    X_test_d = method.transform(X_test)
    time_dtest = time.time() - ts
    print(datetime.datetime.now(), 'Data transformed in %.2f' % (time_dtest + time_dtrain))

    print(datetime.datetime.now(), 'Evaluating shapelets')
    eval_clf_list = _classification(X_train_d, y_train, X_test_d, y_test, n_classes)
    for eval_clf in eval_clf_list:
        eval_clf['dataset'] = dataset_name
        eval_clf['method'] = method_name
        eval_clf['n_shapelets'] = n_sh
        eval_clf['window_sizes'] = ws
        eval_clf['time_train'] = time_train
        eval_clf['time_dtrain'] = time_dtrain
        eval_clf['time_dtest'] = time_dtest

    print(datetime.datetime.now(), 'Storing evaluation')
    store_result(eval_clf_list, method_name)
    return 0


def main():
    method = 'learning' # TO BE RUN

    for dataset in ts_datasets:
        # dataset = 'phalanges'
        print(datetime.datetime.now(), 'Dataset: %s' % dataset)

        D = get_dataset(dataset, path_dataset, normalize)

        run_experiment(D, dataset, method)
        print('')
        # break


if __name__ == "__main__":
    main()
