import platform

if 'Linux' in platform.platform():
    path = '/home/riccardo/Documenti/ExplainingPairwiseLearning/code/'
else:
    path = '/Users/riccardo/Documents/ExplainingPairwiseLearning/code/'

path_dataset = path + 'dataset/'
path_models = path + 'model/'
path_eval = path + 'eval/'

random_state = None
test_size = 0.3
normalize = 'standard'

ts_methods = ['grabocka', 'random', 'random_rnd', 'learning', 'dash', 'dash_rnd', 'dash_md', 'dash_rnd_md']

ts_datasets = ['gunpoint', 'italypower', 'arrowhead', 'ecg200', 'phalanges']  #, 'electricdevices']

tab_datasets = ['wdbc', 'diabetes', 'ctg', 'ionoshpere', 'parkinsons', 'sonar', 'vehicle',
                #'wine', 'winer', 'winew', 'avila'
                ]

txt_datasets = ['20newsgroups', 'imdb']

img_datasets = ['mnist', 'fashion_mnist', 'cifar10']

n_shapelets = 10   # parametro da testare per vedere come cambiano i risultati (meglio se stabili)

n_clusters = 5     # parametro da testare per vedere come cambiano i risultati (meglio se stabili)



from sklearn import tree