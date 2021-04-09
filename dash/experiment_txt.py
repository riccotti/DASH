
from dash.dash_txt import IDACS_TXT_TFIDF
from dash.experiment_evaluate import *
from dash.experiment_evaluate import _classification


def run_random(n_shapelets, window_sizes):
    method = IDACS_TXT_TFIDF(window_sizes=window_sizes, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering=None, train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_random_rnd(n_shapelets, window_sizes):
    method = IDACS_TXT_TFIDF(window_sizes=window_sizes, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering=None, train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash(n_shapelets, window_sizes):
    method = IDACS_TXT_TFIDF(window_sizes=window_sizes, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmeans', train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash_med(n_shapelets, window_sizes):
    method = IDACS_TXT_TFIDF(window_sizes=window_sizes, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmedoids', train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash_rnd(n_shapelets, window_sizes):
    method = IDACS_TXT_TFIDF(window_sizes=window_sizes, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmeans', train_set='sample',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash_rnd_med(n_shapelets, window_sizes):
    method = IDACS_TXT_TFIDF(window_sizes=window_sizes,  n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmedoids', train_set='sample',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_experiment(D, dataset_name, method_name):
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    n_classes = D['n_classes']
    window_sizes = [1, 2]

    if method_name == 'dash':
        method, n_sh, ws = run_dash(n_shapelets, window_sizes)

    elif method_name == 'dash_md':
        method, n_sh, ws = run_dash_med(n_shapelets, window_sizes)

    elif method_name == 'dash_rnd':
        method, n_sh, ws = run_dash_rnd(n_shapelets, window_sizes)

    elif method_name == 'dash_rnd_md':
        method, n_sh, ws = run_dash_rnd_med(n_shapelets, window_sizes)

    elif method_name == 'random':
        method, n_sh, ws = run_random(n_shapelets, window_sizes)

    elif method_name == 'random_rnd':
        method, n_sh, ws = run_random_rnd(n_shapelets, window_sizes)

    else:
        raise ValueError('Unknown method %s' % method_name)

    ts = time.time()
    if 'random' in method_name:
        method.fit(X_train, y_train, sample_size=0.01)
    else:
        method.fit(X_train, y_train)

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
    store_result(eval_clf_list, 'txt2_' + method_name)
    return 0


def main():
    method = 'random_rnd'

    for dataset in txt_datasets[1:]:
        print(datetime.datetime.now(), 'Dataset: %s' % dataset)

        if dataset == '20newsgroups':
            categories = ['alt.atheism', 'talk.religion.misc']
        else:
            categories = None

        D = get_dataset(dataset, path_dataset, normalize, categories=categories)

        run_experiment(D, dataset, method)
        print('')
        # break


if __name__ == "__main__":
    main()
