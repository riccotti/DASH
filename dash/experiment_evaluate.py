import json
import time
import datetime

from dash.util import *
from dash.config import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report


def _evaluate_classification(clf, clf_name, X_train_d, y_train, X_test_d, y_test):
    ts = time.time()
    y_pred_train = clf.predict(X_train_d)
    pred_train_time = time.time() - ts
    ts = time.time()
    y_pred_test = clf.predict(X_test_d)
    pred_test_time = time.time() - ts

    eval_dict = {
        'clf': clf_name,
        'train_acc': accuracy_score(y_train, y_pred_train),
        'test_acc': accuracy_score(y_test, y_pred_test),
        'pred_train_time': pred_train_time,
        'pred_test_time': pred_test_time,
    }

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    for k, v in train_report.items():
        if not isinstance(v, dict):
            continue
        for k1, v1 in v.items():
            eval_dict['train_%s_%s' % (k, k1)] = v1

    test_report = classification_report(y_test, y_pred_test, output_dict=True)
    for k, v in test_report.items():
        if not isinstance(v, dict):
            continue
        for k1, v1 in v.items():
            eval_dict['test_%s_%s' % (k, k1)] = v1

    return eval_dict


def _classification(X_train, y_train, X_test, y_test, n_classes):

    train_times = dict()

    ts = time.time()
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    time_train = time.time() - ts
    train_times['dt'] = time_train

    ts = time.time()
    max_depth = max(4, int(np.ceil(np.log2(n_classes))))
    dt_md = DecisionTreeClassifier(max_depth=max_depth)
    dt_md.fit(X_train, y_train)
    time_train = time.time() - ts
    train_times['dt_md'] = time_train

    ts = time.time()
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(X_train, y_train)
    time_train = time.time() - ts
    train_times['knn1'] = time_train

    ts = time.time()
    X_s, _, y_s, _ = train_test_split(X_train, y_train, train_size=max(10, n_classes), random_state=None, stratify=y_train)
    knn1_s = KNeighborsClassifier(n_neighbors=1)
    knn1_s.fit(X_s, y_s)
    time_train = time.time() - ts
    train_times['knn1_s'] = time_train

    ts = time.time()
    X_o, _, y_o, _ = train_test_split(X_train, y_train, train_size=n_classes, random_state=None, stratify=y_train)
    knn1_o = KNeighborsClassifier(n_neighbors=1)
    knn1_o.fit(X_o, y_o)
    time_train = time.time() - ts
    train_times['knn1_o'] = time_train

    ts = time.time()
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    time_train = time.time() - ts
    train_times['lr'] = time_train

    clf_list = [dt, dt_md, knn1_s, knn1_o, lr]
    clf_names = ['dt', 'dt_md', 'knn1', 'knn1_s', 'knn1_o', 'lr']

    # clf_list = [dt, dt_md, knn1_s, knn1_o, lr]
    # clf_names = ['dt', 'dt_md', 'knn1_s', 'knn1_o', 'lr']

    eval_clf_list = list()
    for clf, clf_name in zip(clf_list, clf_names):
        eval_clf = _evaluate_classification(clf, clf_name, X_train, y_train, X_test, y_test)
        eval_clf['fit_train_time'] = train_times[clf_name]
        eval_clf_list.append(eval_clf)

    return eval_clf_list


def store_result(eval_clf_list, method_name):
    fout = open(path_eval + '%s.json' % method_name, 'alphabet_size')
    for eval_dict in eval_clf_list:
        json_str = ('%s\n' % json.dumps(eval_dict))
        fout.write(json_str)
    fout.close()
