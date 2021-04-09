
import matplotlib.pyplot as plt

from dash.util import *
from dash.config import *

from pyts.transformation import ShapeletTransform

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


def main():

    # dataset = 'italypower'  # ts_datasets[0]
    # window_sizes = [2, 4, 6]
    # window_steps = [1, 1, 1]
    # n_shapelets = 10
    #
    # dataset = 'gunpoint'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 5, 10]
    # n_shapelets = 5
    #
    # dataset = 'arrowhead'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 5, 10]
    # n_shapelets = 5
    #
    # dataset = 'ecg200'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 2, 5]
    # n_shapelets = 5
    #
    # dataset = 'ecg5000'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 2, 5]
    # n_shapelets = 5
    #
    # dataset = 'electricdevices'  # ts_datasets[0]
    # window_sizes = [5, 10, 20]
    # window_steps = [1, 2, 5]
    # n_shapelets = 5
    #
    dataset = 'phalanges'  # ts_datasets[0]
    window_sizes = [5, 10, 20]
    window_steps = [1, 2, 5]
    n_shapelets = 5

    D = get_dataset(dataset, path_dataset, normalize)
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    n_classes = D['n_classes']

    st = ShapeletTransform(n_shapelets=n_shapelets, window_sizes=window_sizes,
                           window_steps=window_steps, n_jobs=-1, verbose=1)
    st.fit(X_train, y_train)

    for s in st.shapelets_:
        plt.plot(s)
    plt.show()

    X_train_d = st.transform(X_train)
    X_test_d = st.transform(X_test)

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
