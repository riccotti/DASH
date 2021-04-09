
from dash.dash_tab import IDACS_TAB


class IDACS_TXT_TFIDF(IDACS_TAB):

    def __init__(self, window_sizes, n_shapelets=1, n_clusters=2, max_comb=10000, apriori_like=True, top_n=1000,
                 clustering='kmeans', train_set='sample', random_state=None, n_jobs=-1, verbose=None):
        super().__init__(window_sizes, n_shapelets, n_clusters, max_comb, apriori_like, top_n,
                         clustering, train_set, random_state, n_jobs, verbose)

