import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold           #it is for cross-validation
from sklearn.model_selection import GridSearchCV    #it is used for all of the 5ML model
from sklearn.feature_selection import RFE    #using recursive feature elimination
import seaborn as sns     
import constants

def plot_feature_importance(importance, names, model_name: str):    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data) #fi for feature importance

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title('MLR FEATURE COEFFICIENTS')
    plt.xlabel('FEATURE COEFFICIENTS')
    plt.ylabel('FEATURE NAMES')
    # plt.savefig(f"results/{model_name} FEATURE COEFFICIENTS.png")
    plt.show()

def train_model(X_train, y_train):
    # step-1: create a cross-validation scheme
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100) #random state is used for getting the same results after shuffle.  #randomly shuffles the data, 

    # step-2: specify range of hyperparameters to tune
    hyper_params = {'n_features_to_select': range(1, 9)}

    # step-3: perform grid search
    # 3.1 specify model
    mlrReg = LinearRegression()
    rfe = RFE(mlrReg)

    # 3.2 call GridSearchCV()
    model_cv = GridSearchCV(estimator = rfe,
                            param_grid = hyper_params,
                            scoring = constants.scoring,
                            refit = 'r2',
                            cv = folds,
                            verbose = constants.print_training_logs,
                            return_train_score=True)

    # fit the model
    model_cv.fit(X_train, y_train)
    return model_cv

# plotting cv results
def create_plot(cv_results, model_cv, model_name: str):
    
    plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_r2"], linewidth=2)
    plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_r2"], linewidth=2)
    plt.xlabel('number of features')
    plt.ylabel('r-squared')
    plt.title("MLR - Optimal number of features")
    plt.legend(['test score', 'train score'], loc='upper left')
    #plt.savefig(f"results/{model_name} - Optimal number of features.png")
    plt.show()