
import matplotlib.pyplot as plt
import time

from dash.util import *
from dash.config import *
from dash.dash_tab import IDACS_TAB

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


def main():
    dataset = 'wdbc'
    window_sizes = [1, 2, 4]
    n_shapelets = 10
    n_clusters = 5

    D = get_dataset(dataset, path_dataset, normalize)
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    n_features = D['n_features']
    feature_names = D['feature_names']
    n_classes = D['n_classes']
    class_values = D['class_values']

    clustering = 'kmeans'
    random_state = None
    n_jobs = -1
    verbose = 1

    dase = IDACS_TAB(window_sizes=window_sizes, n_shapelets=n_shapelets,
                     n_clusters=n_clusters, clustering=clustering, train_set='all',
                     random_state=random_state, n_jobs=n_jobs, verbose=verbose)

    ts = time.time()
    dase.fit(X_train, y_train, sample_size=1000)
    te = time.time()
    print('runtime', (te-ts))

    # print(dase.shapelets_)
    # print(dase.indices_)
    # print(dase.scores_)

    for s, i, l in zip(dase.shapelets_, dase.indices_, dase.scores_):
        print(', '.join(['%s = %s' % (feature_names[iv], sv) for sv, iv in zip(s, i)]), l)

    # locs = dase.locate(X_train[:10])
    # print(locs)

    X_train_d = dase.transform(X_train)
    X_test_d = dase.transform(X_test)

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
