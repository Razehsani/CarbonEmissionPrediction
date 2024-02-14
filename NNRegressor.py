from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import constants

def train_model(X_train, y_train):
    # step-1: create a cross-validation scheme
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100) #random state is used for getting the same results after shuffle.  #randomly shuffles the data, 

    # step-2: specify range of hyperparameters to tune
    hyper_params = {"hidden_layer_sizes": [(8,),(8,),(8,)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": [0.00005, 0.0005]}

    # step-3: perform grid search
    # 3.1 specify model
    mlpReg = MLPRegressor()

    # 3.2 call GridSearchCV()
    model_cv = GridSearchCV(estimator = mlpReg, 
                            param_grid = hyper_params, 
                            scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'neg_mean_squared_log_error'], #strategy to evaluate the performance of the cross-validation model  
                            refit = 'r2',
                            cv=folds,
                            verbose = constants.print_training_logs,
                            return_train_score=True)      

    # fit the model
    model_cv.fit(X_train, y_train)
    return model_cv