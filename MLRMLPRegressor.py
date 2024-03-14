from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import constants

def train_model(X_train, y_train, feature_names):
    
    optimized_train_set, optimized_indices = use_optimized_features(X_train, y_train, feature_names);
    
    # step-1: create a cross-validation scheme
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100) #random state is used for getting the same results after shuffle.  #randomly shuffles the data, 

    # step-2: specify range of hyperparameters to tune
    hyper_params = {"hidden_layer_sizes": [(8,),(8,),(8,)],
                    "activation": ["identity", "logistic", "tanh", "relu"],
                    "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": [0.00005, 0.0005]}

    # step-3: perform grid search
    # 3.1 specify model
    mlrmlpReg = MLPRegressor()

    # 3.2 call GridSearchCV()
    model_cv = GridSearchCV(estimator = mlrmlpReg, 
                            param_grid = hyper_params, 
                            scoring = constants.scoring,
                            refit = 'r2',
                            cv=folds,
                            verbose = constants.print_training_logs,
                            return_train_score=True)      

    # fit the model
    model_cv.fit(optimized_train_set, y_train)
    return model_cv, optimized_indices

def use_optimized_features(x_train, y_train, feature_names):
    lm = LinearRegression()
    rfe = RFE(lm)
    rfe = rfe.fit(x_train, y_train)
    print(rfe.get_feature_names_out(feature_names))
    optimized_feature_names = list(rfe.get_feature_names_out(feature_names))
    indices = [i for i, x in enumerate(feature_names) if x in list(optimized_feature_names)]
    return (x_train[:, indices], indices)