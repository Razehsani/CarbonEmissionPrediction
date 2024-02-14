import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import constants

def plot_feature_importance(importance, names, model_name: str):    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title('RANDOM FOREST FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    # plt.savefig(f"results/{model_name} FEATURE IMPORTANCE.png")
    plt.show()
    

def train_model(X_train, y_train):
    # step-1: create a cross-validation scheme
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100) #random state is used for getting the same results after shuffle.  #randomly shuffles the data, 

    # step-2: specify range of hyperparameters to tune
    hyper_params = [{'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2'],
            'max_depth': list(range(1, 10)),
            'random_state': [18]}]

    # step-3: perform grid search
    # 3.1 specify model
    lm = RandomForestRegressor()

    # 3.2 call GridSearchCV()
    model_cv = GridSearchCV(estimator = lm, 
                            param_grid = hyper_params, 
                            scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'neg_mean_squared_log_error'], #strategy to evaluate the performance of the cross-validation model  
                            refit = 'r2',
                            cv=folds,
                            verbose = constants.print_training_logs,
                            return_train_score=True)      

    # fit the model
    model_cv.fit(X_train, y_train)
    return model_cv


# plotting cv results
def create_plot(cv_results, model_name: str):
    plt.plot(cv_results["param_max_depth"], cv_results["mean_test_score"], linewidth=5)
    plt.plot(cv_results["param_max_depth"], cv_results["mean_train_score"], linewidth=5)
    plt.xlabel('max-depth')
    plt.ylabel('r-squared')
    plt.title("RF - Optimal max-depth")
    plt.legend(['test score', 'train score'], loc='upper left')
    # plt.savefig(f"results/{model_name} - Optimal max-depth.png")
    plt.show()