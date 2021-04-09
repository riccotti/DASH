
import matplotlib.pyplot as plt

from dash.util import *
from dash.config import *

from tslearn.shapelets import ShapeletModel
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


def main():

    # dataset = 'italypower'  # ts_datasets[0]
    # window_sizes = [2, 4, 6]
    # window_steps = [1, 1, 1]
    # n_shapelets = 10
    # n_clusters = 5
    #
    # dataset = 'gunpoint'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 5, 10]
    # n_shapelets = 5
    # n_clusters = 3
    #
    # dataset = 'arrowhead'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 5, 10]
    # n_shapelets = 5
    # n_clusters = 3
    #
    # dataset = 'ecg200'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 2, 5]
    # n_shapelets = 5
    # n_clusters = 3
    #
    # dataset = 'ecg5000'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 2, 5]
    # n_shapelets = 5
    # n_clusters = 2
    #
    # dataset = 'electricdevices'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 2, 5]
    # n_shapelets = 5
    # n_clusters = 5
    #
    dataset = 'phalanges'  # ts_datasets[0]
    window_sizes = [5, 10, 20]
    window_steps = [1, 2, 5]
    n_shapelets = 5
    n_clusters = 5

    D = get_dataset(dataset, path_dataset, normalize)
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    n_classes = D['n_classes']

    # Set the number of shapelets per size as done in the original paper
    n_ts, ts_sz = X_train.shape
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                           ts_sz=ts_sz,
                                                           n_classes=n_classes,
                                                           l=0.1,
                                                           r=1)

    print('n_ts', n_ts)
    print('ts_sz', ts_sz)
    print('n_classes', n_classes)
    print('shapelet_sizes', shapelet_sizes)

    shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes, optimizer='sgd',
                            weight_regularizer=.01, max_iter=1000, verbose=1)
    shp_clf.fit(X_train, y_train)

    for s in shp_clf.shapelets_:
        plt.plot(s)
    plt.show()

    X_train_d = shp_clf.transform(X_train)
    X_test_d = shp_clf.transform(X_test)

    print(' --- DT --- ')

    max_depth = max(4, int(np.ceil(np.log2(n_classes))))

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train_d, y_train)

    print(accuracy_score(y_train, clf.predict(X_train_d)))
    print(accuracy_score(y_test, clf.predict(X_test_d)))

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    print(accuracy_score(y_train, clf.predict(X_train)))
    print(accuracy_score(y_test, clf.predict(X_test)))

    print(' --- 1NN --- ')

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train_d, y_train)

    print(accuracy_score(y_train, clf.predict(X_train_d)))
    print(accuracy_score(y_test, clf.predict(X_test_d)))

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)

    print(accuracy_score(y_train, clf.predict(X_train)))
    print(accuracy_score(y_test, clf.predict(X_test)))

    print(' --- LR --- ')

    clf = LogisticRegression()
    clf.fit(X_train_d, y_train)

    print(accuracy_score(y_train, clf.predict(X_train_d)))
    print(accuracy_score(y_test, clf.predict(X_test_d)))

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    print(accuracy_score(y_train, clf.predict(X_train)))
    print(accuracy_score(y_test, clf.predict(X_test)))


if __name__ == "__main__":
    main()
