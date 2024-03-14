import constants
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_dataset(dataset_name: str, cft_evaluate: bool, experiment_type: str):
    # Read dataset
    if (dataset_name.endswith("csv")):
        dataset = pd.read_csv(dataset_name)
    elif (dataset_name.endswith("xlsx")):
        dataset = pd.read_excel(dataset_name)

    dataset = dataset.fillna(0) #impute #transformation

    # Normalizing [Cultivated area (hectares)] feature #droping cultivated area # transformation
    if ("Cultivated area (hectares)" in dataset.columns):
        for column in dataset.columns:
            if (column not in constants.excluded_normalization_columns):
                dataset[column] = dataset[column] / \
                    dataset["Cultivated area (hectares)"]

    else:
        for column in dataset.columns:
            if (column not in constants.excluded_normalization_columns):
                dataset[column] = dataset[column] / dataset[0]

    # Constants are extracted manually from the dataset
    independent_variables = constants.independant_variables_potato if experiment_type == "potato" else constants.independant_variables_other_crops
    dependent_variable = constants.dependant_variable_potato if experiment_type == "potato" else constants.dependant_variable_other_crops
    cft_dependant_variable = constants.cft_dependant_variable_potato if experiment_type == "potato" else constants.cft_dependant_variable_other_crops
    
    X = dataset.iloc[:, independent_variables].values
    y = dataset.iloc[:, dependent_variable].values if cft_evaluate == False else dataset.iloc[:, cft_dependant_variable].values

    # Scaling input and output variables features as a transformation
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    X_scaled = scaler.fit_transform(X)
    y = pd.DataFrame(y_scaled)
    X = pd.DataFrame(X_scaled)
    return (dataset, X, y)
