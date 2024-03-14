import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import GridSearchCV
import pickle
from pathlib import Path

import constants           #I imported all of our required modules
import KNN as knn
import Lasso as lasso
import MLR as mlr
import NNRegressor as nnreg
import RF as rf
import MLRMLPRegressor as mlrmlp
import utils
import csv

# we have some functions here. every of these functions do special things,after definition of functions 2functions will
# be called,initialize() and run_experiment().

final_results = []  #this is a parameter for 

def write_results_to_csv(experiment_name):
    file_path = f"all results/{experiment_name}/results/{constants.scores_output_csv_file}"
    with open(file_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        for row in final_results:
            writer.writerow(row)


def initialize():
    plt.figure(figsize=(16, 6))     #dimention of printed diagrams(by experience I came to this result this is a good size)
    font = {'family' : 'Times new roman',
            'size'   : 12}
    mpl.rc('font', **font)
    if (not (Path.exists(Path("all results")))): #if this folder won't exist, go and create it.
        Path.mkdir(Path("all results"), exist_ok=True) #mkdir for make directory #exist_ok, to not getting error if we had folder

    for experiment in constants.experiments:   #for every experiment we defined in constants.py we get the title of experiment.
                                                
        experiment_name = experiment["title"]
        create_output_directories(experiment_name) #create output folders # we have results folder and pickles folder in all results
    
        
        if(experiment["generate_cft_result"] == True):  #for every experiment, it asks we want the results of cool farm tool or not?
            experiment_name = experiment["title"] + " - cool farm tool"
            create_output_directories(experiment_name)
        

def create_output_directories(experiment_name: str):     # example.all results-->Iranian potato only-->pickles & results
    if (not (Path.exists(Path(f"all results/{experiment_name}")))): #this function will bulid these folders 
        Path.mkdir(Path(f"all results/{experiment_name}"), exist_ok=True) #  will build (results.txt) and (scores.csv)
            
    if (not (Path.exists(Path(f"all results/{experiment_name}/results")))):
        Path.mkdir(Path(f"all results/{experiment_name}/results"), exist_ok=True) 
        
    if (not (Path.exists(Path(f"all results/{experiment_name}/pickles")))):
        Path.mkdir(Path(f"all results/{experiment_name}/pickles"), exist_ok=True)

    with open(f"all results/{experiment_name}/results/{constants.scores_output_file_name}", "w") as f:
        f.close()
        
    with open(f"all results/{experiment_name}/results/{constants.scores_output_csv_file}", "w") as f:
        f.close()


def read_datasets_and_prepare_model(file_name, datasets_dic, experiment_name, cft_evaluate, experiment_type):
    train_test_datasets = []  
    X_train = np.empty([1, 1])
    y_train = np.empty([1, 1])

    if (Path.exists(Path(f"all results/{experiment_name}/pickles/{file_name}"))):  #if already loaded or not
        train_test_datasets = pickle.load(
            open(f"all results/{experiment_name}/pickles/{file_name}", 'rb'))

        for index in range(len(datasets_dic)):
            if (datasets_dic[index]["use_in_train"]):
                if (index == 0):
                    X_train = train_test_datasets[index][2].values
                    y_train = train_test_datasets[index][4].values
                else:
                    X_train = np.concatenate(
                        (X_train, train_test_datasets[index][2].values))
                    y_train = np.concatenate(
                        (y_train, train_test_datasets[index][4].values))

        y_train = np.ravel(y_train)
        return (train_test_datasets, X_train, y_train)

    for index in range(len(datasets_dic)):
        trainCropDataset, X_train_crop, y_train_crop = utils.load_dataset(                       
            datasets_dic[index]["train_file_path"], cft_evaluate, experiment_type)      #call utils for train dataset
        testCropDataset, X_test_crop, y_test_crop = utils.load_dataset(
            datasets_dic[index]["test_file_path"], cft_evaluate, experiment_type)       # call utils for test dataset

        train_test_datasets.append(
            [trainCropDataset, testCropDataset, X_train_crop, X_test_crop, y_train_crop, y_test_crop])

        if (datasets_dic[index]["use_in_train"]):
            if (index == 0):
                X_train = X_train_crop.values
                y_train = y_train_crop.values
            else:
                X_train = np.concatenate((X_train, X_train_crop.values))
                y_train = np.concatenate((y_train, y_train_crop.values))

    pickle.dump(train_test_datasets, open(f"all results/{experiment_name}/pickles/{file_name}", 'wb'))
    y_train = np.ravel(y_train)
    return (train_test_datasets, X_train, y_train)  #if we had pickles before, we do not need to load dataset and ztrain again


def create_test_results(model_name: str, model_cv: GridSearchCV, train_test_datasets: list, experiment: dict, experiment_name: str, optimized_features: list = []):
    file_path = f"all results/{experiment_name}/results/{constants.scores_output_file_name}"
    
    with open(file_path, "a") as f:
        f.write(f"#### {model_name} ####\n")
        f.write(f"Best params: {model_cv.best_params_}\n")
        f.write(f"Best score (r2): {model_cv.best_score_}\n")
        final_results.append([model_name, f"{model_cv.best_params_}", model_cv.best_score_])

        X_test_total = np.empty([1, 1])
        y_test_total = np.empty([1, 1])

        accumulator = False

        for i in range(len(experiment["datasets"])):
            details = []
            if (experiment["datasets"][i]["use_in_test"]):

                if (accumulator == False):
                    X_test_total = train_test_datasets[i][3]
                    y_test_total = train_test_datasets[i][5]
                    accumulator = True
                else:
                    X_test_total = np.concatenate(
                        (X_test_total, train_test_datasets[i][3]))
                    y_test_total = np.concatenate(
                        (y_test_total, train_test_datasets[i][5]))

                if(len(optimized_features) > 0):
                    y_pred = model_cv.best_estimator_.predict(
                        train_test_datasets[i][3].iloc[:, optimized_features])
                else:
                    y_pred = model_cv.best_estimator_.predict(
                        train_test_datasets[i][3])
                crop_name = experiment["datasets"][i]["dataset_displayname"]
                details.append(crop_name)
                r2Score = r2_score(train_test_datasets[i][5], y_pred)
                rmse = np.sqrt(np.abs(mean_squared_error(train_test_datasets[i][5], y_pred)))
                mae = mean_absolute_error(train_test_datasets[i][5], y_pred)
                details.append(r2Score)
                details.append(rmse)
                details.append(mae)
                f.write(
                    f"Test score for dataset {crop_name} (r2): {r2Score}\n")
                f.write(
                    f"RMSE for dataset {crop_name}: {rmse}\n")
                f.write(
                    f"MAE for dataset {crop_name}: {mae}\n")

                try:
                    msle = mean_squared_log_error(train_test_datasets[i][5], y_pred)
                    details.append(msle)
                    f.write(
                        f"MSLE for dataset {crop_name}: {msle}\n")
                except Exception as e:
                    details.append("N/A")
                    f.write(
                        f"MSLE is not available for {crop_name}: {e}\n")
                f.write("\n")
                final_results.append(details)

        if (len(experiment["datasets"]) > 1):
            details = []
            if(len(optimized_features) > 0):
                y_pred_total = model_cv.best_estimator_.predict(X_test_total[:, optimized_features])
            else:
                y_pred_total = model_cv.best_estimator_.predict(X_test_total)
            
            r2Score_total = r2_score(y_test_total, y_pred_total)
            rmse_total = np.sqrt(np.abs(mean_squared_error(y_test_total, y_pred_total)))
            mae_total = mean_absolute_error(y_test_total, y_pred_total)
            
            details.append("Total scores")
            details.append(r2Score_total)
            details.append(rmse_total)
            details.append(mae_total)
            
            f.write(
                f"Total test score (r2): {r2Score_total}\n")
            f.write(
                f"Total RMSE: {rmse_total}\n")
            f.write(
                f"Total MAE: {mae_total}\n")

            try:
                msle_total = mean_squared_log_error(y_test_total, y_pred_total)
                details.append(msle_total)
                f.write(
                    f"Total MSLE: {msle_total}\n")
            except Exception as e:
                details.append("N/A")
                f.write(
                    f"Total MSLE is not available: {e}\n")
                
            final_results.append(details)

        f.write("\n\n\n\n")
        f.close()

    cv_results = pd.DataFrame(model_cv.cv_results_)
    return cv_results


def load_or_train_model(file_name: str, experiment_name: str, X_train, y_train, feature_names = []):
    optimized_indices = None
    file_path = f"all results/{experiment_name}/pickles/{file_name}"
    if (Path.exists(Path(file_path))):
        loaded_model = pickle.load(open(file_path, 'rb'))
        return loaded_model
    elif (file_name.startswith("KNN")):
        model_cv = knn.train_model(X_train, y_train)
    elif (file_name.startswith("Lasso")):
        model_cv = lasso.train_model(X_train, y_train)
    elif (file_name.startswith("MLR")):
        model_cv = mlr.train_model(X_train, y_train)
    elif (file_name.startswith("NNRegressor")):
        model_cv = nnreg.train_model(X_train, y_train)
    elif (file_name.startswith("RF")):
        model_cv = rf.train_model(X_train, y_train)
    elif (file_name.startswith("LR-MLP")):
        model_cv, optimized_indices = mlrmlp.train_model(X_train, y_train, feature_names)

    # save the model to disk
    if(optimized_indices is None):
        pickle.dump(model_cv, open(file_path, 'wb'))
        return model_cv
    else:
        pickle.dump((model_cv, optimized_indices), open(file_path, 'wb'))
        return (model_cv, optimized_indices)


def create_and_run_all_models(X_train: np.ndarray, y_train: np.ndarray, train_test_datasets: list, augmented: bool, experiment: dict, experiment_name: str):

    knn_name = "KNN_aug" if augmented else "KNN"
    lasso_name = "Lasso_aug" if augmented else "Lasso"
    mlr_name = "MLR_aug" if augmented else "MLR"
    nnreg_name = "NNRegressor_aug" if augmented else "NNRegressor"
    rf_name = "RF_aug" if augmented else "RF"
    lr_mlp_name = "LR-MLP_aug" if augmented else "LR-MLP"
    
    independent_variables = constants.independant_variables_potato if experiment["experiment_type"] == "potato" else constants.independant_variables_other_crops
    all_independent_feature_names = train_test_datasets[0][0].columns[independent_variables]
    final_results.clear()

    # KNN
    print("Evaluating KNN model...")
    knn_model = load_or_train_model(f"{knn_name}.sav", experiment_name, X_train, y_train)
    knn_cv_results = create_test_results(
        knn_name, knn_model, train_test_datasets, experiment, experiment_name)

    # Lasso
    print("Evaluating Lasso model...")
    lasso_model = load_or_train_model(f"{lasso_name}.sav", experiment_name, X_train, y_train)
    lasso_cv_results = create_test_results(
        lasso_name, lasso_model, train_test_datasets, experiment, experiment_name)

    # MLR
    print("Evaluating MLR model...")
    mlr_model = load_or_train_model(f"{mlr_name}.sav", experiment_name, X_train, y_train)
    mlr_cv_results = create_test_results(
        mlr_name, mlr_model, train_test_datasets, experiment, experiment_name)

    # NNRegressor
    print("Evaluating NNRegressor model...")
    nnreg_model = load_or_train_model(f"{nnreg_name}.sav", experiment_name, X_train, y_train)
    nnreg_cv_results = create_test_results(
        nnreg_name, nnreg_model, train_test_datasets, experiment, experiment_name)

    # RF
    print("Evaluating RF model...")
    rf_model = load_or_train_model(f"{rf_name}.sav", experiment_name, X_train, y_train)
    rf_cv_results = create_test_results(rf_name, rf_model, train_test_datasets, experiment, experiment_name)

    # LR-MLP
    print("Evaluating LR-MLP model...")
    lr_mlp_model, optimized_lr_mlp_features = load_or_train_model(f"{lr_mlp_name}.sav", experiment_name, X_train, y_train,
                                                                  all_independent_feature_names)
    lr_mlp_cv_results = create_test_results(
        lr_mlp_name, lr_mlp_model, train_test_datasets, experiment, experiment_name, optimized_lr_mlp_features)
    
    # Create plots
    knn.create_plot(knn_cv_results, knn_model, knn_name)
    
    lasso.create_plot(lasso_cv_results, lasso_model, lasso_name)
    lasso.plot_feature_importance(lasso_model.best_estimator_.coef_,
                                  all_independent_feature_names, lasso_name)
    
    mlr.create_plot(mlr_cv_results, mlr_model, mlr_name)
    mlr.plot_feature_importance(mlr_model.best_estimator_.estimator_.coef_[
                                0], all_independent_feature_names[mlr_model.best_estimator_.support_], mlr_name)
    
    rf.create_plot(rf_cv_results, rf_model, rf_name)
    rf.plot_feature_importance(rf_model.best_estimator_.feature_importances_, all_independent_feature_names,
                               rf_name)

    write_results_to_csv(experiment_name)


def run_experiments(): 
    for experiment in constants.experiments: # this specific function is said; for every experiment in constants.py run some models
        experiment_name = experiment["title"] #first grab title
        print(f"\n\n\nRun model for experiment {experiment_name}") #run models
        #experiment type: potato or other we have four state: manual_normal , manual_aug, cft_normal, cft_aug
        
        #this function gives us 3output(train_test_datasets, X_train, y_train) #read_datasets_and_prepare_model will be called
        train_test_datasets, X_train, y_train = read_datasets_and_prepare_model(
            constants.train_test_datasets_file_name, experiment["datasets"], experiment_name, False, experiment["experiment_type"])
        
        #input parameters are dataset file_name,,datasets,experiment_name,experiment_type(from constants file)
        create_and_run_all_models(X_train, y_train, train_test_datasets, False, experiment, experiment_name)

        if (len(experiment["aug_datasets"]) > 0):
            print(f"\n\n\nRun model for experiment {experiment_name} - augmented")
            train_test_datasets_aug, X_train_aug, y_train_aug = read_datasets_and_prepare_model(
                constants.train_test_datasets_aug_file_name, experiment["aug_datasets"], experiment_name, False, experiment["experiment_type"])

            X_train = np.concatenate((X_train, X_train_aug))
            y_train = np.concatenate((y_train, y_train_aug))

            create_and_run_all_models(X_train, y_train, train_test_datasets, True, experiment, experiment_name)

        if(experiment["generate_cft_result"] == True):
            experiment_name = experiment["title"] + " - cool farm tool"
            print(f"\n\n\nRun model for experiment {experiment_name}")
            
            train_test_datasets, X_train, y_train = read_datasets_and_prepare_model(
                constants.train_test_datasets_file_name, experiment["datasets"], experiment_name, True, experiment["experiment_type"])

            create_and_run_all_models(X_train, y_train, train_test_datasets, False, experiment, experiment_name)

            if (len(experiment["aug_datasets"]) > 0):
                print(f"\n\n\nRun model for experiment {experiment_name} - augmented")
                train_test_datasets_aug, X_train_aug, y_train_aug = read_datasets_and_prepare_model(
                    constants.train_test_datasets_aug_file_name, experiment["aug_datasets"], experiment_name, True, experiment["experiment_type"])

                X_train = np.concatenate((X_train, X_train_aug))
                y_train = np.concatenate((y_train, y_train_aug))

                create_and_run_all_models(X_train, y_train, train_test_datasets, True, experiment, experiment_name)


initialize() #this function is for setting the parameters, creating the required folders. #in the flow of the code initialize()
             #will be called first, and initialize will call create_output_directory,

run_experiments(); #for every experiment that defined in the constants.py do some work,