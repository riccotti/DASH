
import matplotlib.pyplot as plt

from dash.util import *
from dash.config import *
from dash.dash_img import IDACS_IMG

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


def main():

    dataset = 'mnist'  # ts_datasets[0]
    window_sizes = [(7, 7), (14, 14)]
    window_steps = [(7, 7), (14, 14)]
    n_shapelets = 10
    n_clusters = 5

    D = get_dataset(dataset, path_dataset, normalize, categories=[0, 1, 2, 3])
    X_train, y_train, X_test, y_test = D['X_train'], D['y_train'], D['X_test'], D['y_test']
    n_classes = D['n_classes']
    class_values = D['class_values']
    w, h = D['word_size'], D['h']
    # print(X_train.shape)
    # print(n_classes)
    # print(class_values)

    # plt.matshow(X_train[0] * 255, cmap='gray')
    # plt.show()
    # return -1

    clustering = 'kmeans'
    random_state = None
    n_jobs = -1
    verbose = 1

    dase = IDACS_IMG(window_sizes=window_sizes, window_steps=window_steps, n_shapelets=n_shapelets,
                    n_clusters=n_clusters, clustering=clustering, train_set='sample',
                    random_state=random_state, n_jobs=n_jobs, verbose=verbose)

    dase.fit(X_train, y_train, sample_size=1000)

    # print(dase.shapelets_)
    # print(dase.indices_)
    # print(dase.scores_)

    # for s in dase.shapelets_:
    #     plt.matshow(s * 255, cmap='gray')
    #     plt.show()

    locations = dase.locate(X_test[:10])
    # # print(locations)
    # # print(len(locations[0]))
    #
    test_idx = 0
    # plt.matshow(X_test[test_idx] * 255, cmap='gray')
    # plt.show()

    ws = np.min(window_steps, axis=0)
    for i in range(n_shapelets):
        s = dase.shapelets_[i]
        # plt.imshow(s * 255, cmap='gray')
        # plt.show()

        sl = dase.indices_[i][2] - dase.indices_[i][1]
        # print(sl)
        # print('locations', locations[test_idx][i])
        # print('ws', ws)
        # print('sl', sl)
        div_w = (w - sl[0]) // ws[0] + 1
        div_h = (h - sl[1]) // ws[1] + 1
        # print('div', div_w, div_h)
        si_w = (int(locations[test_idx][i]) // div_w)
        si_h = (int(locations[test_idx][i]) % div_h)
        # print(si_w, si_h)
        si_w = (int(locations[test_idx][i]) // div_w) * sl[0]
        si_h = (int(locations[test_idx][i]) % div_h) * sl[1]
        # print(si_w, si_h)
        simg = np.zeros((w, h))
        simg[si_w:si_w+sl[0], si_h:si_h+sl[1]] = s
        # print(np.where(simg > 0))

        plt.imshow(X_test[test_idx] * 255, cmap='gray')
        simg = np.stack((np.zeros((w, h)), simg, np.zeros((w, h))), axis=2)
        # plt.imshow(simg * 255, alpha=0.5)
        plt.imshow(np.ma.masked_where(simg > 0, simg) * 255, alpha=0.7)
        plt.show()
    # plt.show()
    #
    return -1

    X_train_d = dase.transform(X_train)
    X_test_d = dase.transform(X_test)

    s0, s1, s2 = X_train.shape
    s3, _, _ = X_test.shape

    print(' --- DT --- ')

    max_depth = max(4, int(np.ceil(np.log2(n_classes))))

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train_d, y_train)

    print(accuracy_score(y_train, clf.predict(X_train_d)))
    print(accuracy_score(y_test, clf.predict(X_test_d)))

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train.reshape(s0, s1 * s2), y_train)

    print(accuracy_score(y_train, clf.predict(X_train.reshape(s0, s1 * s2))))
    print(accuracy_score(y_test, clf.predict(X_test.reshape(s3, s1 * s2))))

    # print(' --- 1NN --- ')
    #
    # clf = KNeighborsClassifier(n_neighbors=1)
    # clf.fit(X_train, y_train)
    #
    # print(accuracy_score(y_train, clf.predict(X_train)))
    # print(accuracy_score(y_test, clf.predict(X_test)))
    #
    # clf = KNeighborsClassifier(n_neighbors=1)
    # clf.fit(X_train.reshape(s0, s1 * s2)[:1000], y_train[:1000])
    #
    # print(accuracy_score(y_train, clf.predict(X_train.reshape(s0, s1 * s2))))
    # print(accuracy_score(y_test, clf.predict(X_test.reshape(s3, s1 * s2))))
    #
    # print(' --- LR --- ')
    #
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    #
    # print(accuracy_score(y_train, clf.predict(X_train)))
    # print(accuracy_score(y_test, clf.predict(X_test)))
    #
    # clf = LogisticRegression()
    # clf.fit(X_train.reshape(s0, s1 * s2), y_train)
    #
    # print(accuracy_score(y_train, clf.predict(X_train.reshape(s0, s1 * s2))))
    # print(accuracy_score(y_test, clf.predict(X_test.reshape(s3, s1 * s2))))


if __name__ == "__main__":
    main()
