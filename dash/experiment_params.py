import time
import datetime

from dash.config import path_dataset, normalize
from dash.util import get_dataset, datasets
from dash.dash_ts import IDACS_TS
from dash.dash_tab import IDACS_TAB
from dash.dash_txt import IDACS_TXT_TFIDF
from dash.dash_img import IDACS_IMG
from dash.experiment_evaluate import _classification, store_result


def run_dash(n_shapelets, window_sizes, window_steps, n_clusters, random_state):
    method = IDACS_TS(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmeans', train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash_tab(n_shapelets, window_sizes, n_clusters, random_state):
    method = IDACS_TAB(window_sizes=window_sizes, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmeans', train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash_txt(n_shapelets, window_sizes, n_clusters, random_state):
    method = IDACS_TXT_TFIDF(window_sizes=window_sizes, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmeans', train_set='sample',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_dash_img(n_shapelets, window_sizes, window_steps, n_clusters, random_state):
    method = IDACS_IMG(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                      n_clusters=n_clusters, clustering='kmeans', train_set='all',
                      random_state=random_state, n_jobs=-1, verbose=0)
    return method, n_shapelets, window_sizes


def run_experiment(D, dataset_name, method_name, n_shapelets, n_clusters, random_state):
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    n_classes = D['n_classes']
    dataset_type = datasets[dataset_name][1]
    if 'window_sizes' in D:
        window_sizes = D['window_sizes']
        window_steps = D['window_steps']
    else:
        window_sizes = [1, 2, 4] if dataset_type == 'tab' else [1, 2]
        window_steps = [1, 1, 1]

    if method_name == 'dash':
        if dataset_type == 'ts':
            method, n_sh, ws = run_dash(n_shapelets, window_sizes, window_steps, n_clusters, random_state)
        elif dataset_type == 'tab':
            method, n_sh, ws = run_dash_tab(n_shapelets, window_sizes, n_clusters, random_state)
        elif dataset_type == 'txt':
            method, n_sh, ws = run_dash_txt(n_shapelets, window_sizes, n_clusters, random_state)
        elif dataset_type == 'img':
            method, n_sh, ws = run_dash_img(n_shapelets, window_sizes, window_steps, n_clusters, random_state)
        else:
            raise ValueError('Unknown dataset type %s' % dataset_type)
    else:
        raise ValueError('Unknown method %s' % method_name)

    ts = time.time()
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
        eval_clf['n_shapelets'] = n_shapelets
        eval_clf['window_sizes'] = ws
        eval_clf['time_train'] = time_train
        eval_clf['time_dtrain'] = time_dtrain
        eval_clf['time_dtest'] = time_dtest
        eval_clf['n_clusters'] = n_clusters

    print(datetime.datetime.now(), 'Storing evaluation')
    store_result(eval_clf_list, 'params_' + method_name)
    return 0


def main():

    random_state = 0

    for method in ['dash']:  #, 'dash_rnd', 'dash_md', 'dash_rnd_md']:
        # for dataset in ['gunpoint', 'italypower', 'ecg200', 'phalanges',
        #                 'wdbc', 'diabetes', 'ctg', 'mnist', '20newsgroup']:
        for dataset in ['20newsgroups']:

            if dataset == 'mnist':
                categories = [0, 1, 2, 3]
            elif dataset == '20newsgroups':
                categories = ['alt.atheism', 'talk.religion.misc']
            else:
                categories = None

            if dataset == '20newsgroups':
                n_words = 100
            else:
                n_words = None

            D = get_dataset(dataset, path_dataset, normalize, categories=categories, n_words=n_words)
            for n_shapelets in [1, 2, 4, 8, 16, 32]:
                for n_clusters in [1, 2, 4, 8, 16]:
                    print(datetime.datetime.now(),
                          method,
                          dataset,
                          'n_shapelets %s' % n_shapelets,
                          'n_clusters %s' % n_clusters,)
                    run_experiment(D, dataset, method, n_shapelets, n_clusters, random_state)
            print('')
        # break


if __name__ == "__main__":
    main()
