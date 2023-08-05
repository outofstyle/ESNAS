from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def load_surrogate_model(name):

    if name == 'knn':
        h1 = KNeighborsRegressor(n_neighbors=3, p=2)
        h2 = KNeighborsRegressor(n_neighbors=3, p=5)

        h1_temp = KNeighborsRegressor(n_neighbors=3, p=2)
        h2_temp = KNeighborsRegressor(n_neighbors=3, p=5)
        return h1, h2, h1_temp, h2_temp

    if name == 'xgb':
        h1 = xgb.XGBRegressor()
        h2 = xgb.XGBRegressor(learning_rate=0.3, n_estimators=140,
                             max_depth=7, min_child_weight=4,
                             subsample=0.86, colsample_bytree=0.87,eta=0.12,reg_alpha=0.87, reg_lambda=20)
        h1_temp = xgb.XGBRegressor()
        h2_temp = xgb.XGBRegressor(learning_rate=0.3, n_estimators=140,
                                     max_depth=7, min_child_weight=4,
                                     subsample=0.86, colsample_bytree=0.87, eta=0.12, reg_alpha=0.87, reg_lambda=20)
        return h1, h2, h1_temp, h2_temp
    if name == 'svr':
        h1 = svm.SVR(kernel='rbf', degree=3, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        h2 = svm.SVR(kernel='rbf', degree=3, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=190, verbose=False, max_iter=-1)

        h1_temp = svm.SVR(kernel='rbf', degree=3, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1,
                            shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        h2_temp = svm.SVR(kernel='rbf', degree=3, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=190, verbose=False, max_iter=-1)
        return h1, h2, h1_temp, h2_temp
    if name == 'mlp':
        h1 = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(300, 300), random_state=1, max_iter=5000,
                                 activation='logistic')
        h2 = MLPRegressor(alpha=1e-6, hidden_layer_sizes=(200, 200), random_state=1, max_iter=5000,
                                 activation='logistic')
        h1_temp = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(300, 300), random_state=1, max_iter=5000,
                                 activation='logistic')
        h2_temp = MLPRegressor(alpha=1e-6, hidden_layer_sizes=(200, 200), random_state=1, max_iter=5000,
                                 activation='logistic')
        return h1, h2, h1_temp, h2_temp
    if name == 'rf':
        h1 = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                          min_weight_fraction_leaf=0.0, max_features=0.17)
        h2 = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2,
                                          min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=0.1705585)
        h1_temp = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2,
                                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=0.17)
        h2_temp = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2,
                                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=0.1705585)
        return h1, h2, h1_temp, h2_temp

    if name =='knns':
        from surrogate_models import SurrogateModel
        h1 = SurrogateModel('knns')
        h2 = SurrogateModel('knne')
        h1_temp = SurrogateModel('knns')
        h2_temp = SurrogateModel('knne')
        return h1, h2, h1_temp, h2_temp
    if name =='gbdt':
        from lightgbm import LGBMRegressor
        h1 = LGBMRegressor(objective='huber')
        h2 = LGBMRegressor(objective='huber')
        h1_temp = LGBMRegressor(objective='huber')
        h2_temp = LGBMRegressor(objective='huber')
        return h1, h2, h1_temp, h2_temp