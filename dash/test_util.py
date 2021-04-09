
from dash.util import *
from dash.config import *


def main():

    for dataset in datasets:
        if dataset == 'omniglot':
            continue
        print(dataset)
        D = get_dataset(dataset, path_dataset)



        if datasets[dataset][1] == 'tab':
            print('X_train.shape', D['X_train'].shape)
            print('X_test.shape', D['X_test'].shape)
            print('y_train.shape', D['y_train'].shape)
            print('n_features', D['n_features'])
            print('n_classes', D['n_classes'])

        elif datasets[dataset][1] == 'img':
            print('X_train.shape', D['X_train'].shape)
            print('X_test.shape', D['X_test'].shape)
            print('y_train.shape', D['y_train'].shape)
            print('word_size, h', D['word_size'], D['h'])
            print('n_classes', D['n_classes'])

        elif datasets[dataset][1] == 'ts':
            print('X_train.shape', D['X_train'].shape)
            print('X_test.shape', D['X_test'].shape)
            print('y_train.shape', D['y_train'].shape)
            print('n_timestamps', D['n_timestamps'])
            print('n_classes', D['n_classes'])

        elif datasets[dataset][1] == 'txt':
            print('X_train.shape', D['X_train'].shape)
            print('X_test.shape', D['X_test'].shape)
            print('y_train.shape', D['y_train'].shape)
            print('n_terms', D['n_terms'])
            print('n_classes', D['n_classes'])

        print('')


if __name__ == "__main__":
    main()
