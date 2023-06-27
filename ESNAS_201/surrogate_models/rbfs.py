import numpy as np
from surrogate_models.rbf import RBF

class RBFEnsemble:

    def __init__(self, ensemble_size=500, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_models = ensemble_size
        self.name = 'rbf_ensemble'
        self.models = None
        self.features = None

    def fit(self, X, y):
        n, m = X.shape
        features = []
        models = []

        print("Constructing RBF ensemble surrogate model with "
              "sample size = {}, ensemble size = {}".format(n, self.n_models))

        for i in range(self.n_models):

            sample_idx = np.arange(n)
            np.random.shuffle(sample_idx)
            X = X[sample_idx, :]
            y = y[sample_idx]

            feature_idx = np.arange(m)
            np.random.shuffle(feature_idx)

            n_feature = np.random.randint(1, m + 1)
            selected_feature_ids = feature_idx[0:n_feature]
            features.append(selected_feature_ids)

            rbf = RBF(kernel='cubic', tail='linear')
            rbf.fit(X[:, selected_feature_ids], y)
            models.append(rbf)

        self.models = models
        self.features = features
        return self.models

    def predict(self, X):
        assert self.models is not None, "RBF models do not exist, call fit to obtain rbf models first"

        models, features = self.models, self.features
        n, n_tree = len(X), len(models)
        y = np.zeros(n)

        for i in range(n):
            this_test_data = X[i, :]
            predict_this_list = np.zeros(n_tree)

            for j, (rbf, feature) in enumerate(zip(models, features)):
                predict_this_list[j] = rbf.predict(np.array([this_test_data[feature]]))[0]

            predict_this_list = np.sort(predict_this_list)
            predict_this_list = predict_this_list[::-1]
            this_predict = np.mean(predict_this_list)
            y[i] = this_predict

        return y[:, None]