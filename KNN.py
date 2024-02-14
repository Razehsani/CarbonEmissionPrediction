import matplotlib.pyplot as plt
from sklearn.model_selection import KFold  #for example 5fold cross-validation
from sklearn.model_selection import GridSearchCV #for knn that concurrent used for number of k in knn and generating cross-validation #improved knn
from sklearn.neighbors import KNeighborsRegressor #cause I want to use regression in knn
import numpy as np
import constants

def train_model(X_train, y_train) -> GridSearchCV:
    
    # step-1: create a cross-validation scheme
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100) #random state is used for getting the same results after shuffle.  #randomly shuffles the data, 

    # step-2: specify range of hyperparameters to tune
    hyper_params = {'n_neighbors': np.arange(1, 30), # it was 200 in my early versions
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'p': np.arange(1, 5)}

    # step-3: perform grid search
    # 3.1 specify model
    knnReg = KNeighborsRegressor() #this class imported from sklearn.neighbors library

    # 3.2 call GridSearchCV()
    model_cv = GridSearchCV(estimator = knnReg,  #model
                            param_grid = hyper_params, #Range of k
                            scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'neg_mean_squared_log_error'], #strategy to evaluate the performance of the cross-validation model  
                            refit = 'r2',
                            cv = folds, #folds is defined in step-1 #cross validation generator
                            verbose = constants.print_training_logs, # print logs while working
                            return_train_score=True) #It provides additional information about how well the model is fitting the training data in each fold.  
    
    # fit the model
    model_cv.fit(X_train, y_train)
    return model_cv

# plotting cv results
def create_plot(cv_results, model_name:str):
    plt.plot(cv_results["param_n_neighbors"], cv_results["mean_test_score"], linewidth=5)
    plt.plot(cv_results["param_n_neighbors"], cv_results["mean_train_score"], linewidth=5)
    plt.xlabel('number of neighbors')
    plt.ylabel('r-squared')
    plt.title("KNN - Optimal number of neighbors")
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.show()