from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

def train_model(X_train, y_train):
    # step-1: create a cross-validation scheme
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

    # step-2: specify range of hyperparameters to tune
    hyper_params = [{"hidden_layer_sizes": [(8,),(8,),(8,)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": [0.00005, 0.0005]}]

    # step-3: perform grid search
    # 3.1 specify model
    lm = MLPRegressor()

    # 3.2 call GridSearchCV()
    model_cv = GridSearchCV(estimator = lm, 
                            param_grid = hyper_params, 
                            scoring= 'r2', #'neg_root_mean_squared_error'
                            cv = folds, 
                            verbose = 1,
                            return_train_score=True)      

    # fit the model
    model_cv.fit(X_train, y_train)
    return model_cv