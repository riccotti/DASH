
from dash.experiment_evaluate import *
from dash.experiment_evaluate import _classification


def run_experiment(D, dataset, method_name):
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    n_classes = D['n_classes']

    if X_train.ndim > 2:
        s0, s1, s2 = X_train.shape
        X_train = X_train.reshape(s0, s1 * s2)
        s0, s1, s2 = X_test.shape
        X_test = X_test.reshape(s0, s1 * s2)

    print(datetime.datetime.now(), 'Evaluating classifiers')
    eval_clf_list = _classification(X_train, y_train, X_test, y_test, n_classes)
    for eval_clf in eval_clf_list:
        eval_clf['dataset'] = dataset

    print(datetime.datetime.now(), 'Storing evaluation')
    store_result(eval_clf_list, method_name)


def main():

    # tutti fatti

    method_name = 'raw_img_4_%s'

    # for dataset in ts_datasets:
    # for dataset in tab_datasets:
    # for dataset in ['mnist', 'fashion_mnist', 'cifar10']:
    # for dataset in ['imdb']:
    #     print(datetime.datetime.now(), 'Dataset: %s' % dataset)
    #     D = get_dataset(dataset, path_dataset, normalize)
    #     run_experiment(D, dataset, method_name)
    #     print('')

    for dataset in img_datasets:
        for cidx, categories in enumerate([[0, 1],
                           [5, 6],
                           [0, 1, 2, 3],
                           ]):
            print(datetime.datetime.now(), 'Dataset: %s' % dataset, categories)

            D = get_dataset(dataset, path_dataset, normalize, categories=categories)
            run_experiment(D, dataset, method_name % cidx)


if __name__ == "__main__":
    main()
