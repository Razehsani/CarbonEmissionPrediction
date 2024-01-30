import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

def train_model(X_train, y_train) -> GridSearchCV:
    
    # step-1: create a cross-validation scheme
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

    # step-2: specify range of hyperparameters to tune
    hyper_params = [{'n_neighbors': list(range(1, X_train.shape[1] + 1))}]

    # step-3: perform grid search
    # 3.1 specify model
    lm = KNeighborsRegressor()

    # 3.2 call GridSearchCV()
    model_cv = GridSearchCV(estimator = lm, 
                            param_grid = hyper_params, 
                            scoring= 'r2',
                            cv = folds, 
                            verbose = 1,
                            return_train_score=True)

    # fit the model
    model_cv.fit(X_train, y_train)
    return model_cv


# plotting cv results
def create_plot(cv_results, model_name: str):
    plt.plot(cv_results["param_n_neighbors"], cv_results["mean_test_score"], linewidth=5)
    plt.plot(cv_results["param_n_neighbors"], cv_results["mean_train_score"], linewidth=5)
    plt.xlabel('number of neighbors')
    plt.ylabel('r-squared')
    plt.title("KNN - Optimal number of neighbors")
    plt.legend(['test score', 'train score'], loc='upper left')
    # plt.savefig(f"results/{model_name} - Optimal number of neighbors.png")
    plt.show()