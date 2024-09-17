import constants
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_dataset(dataset_name: str, cft_evaluate: bool, experiment_type: str):
    # Read dataset
    if (dataset_name.endswith("csv")):
        dataset = pd.read_csv(dataset_name)
    elif (dataset_name.endswith("xlsx")):
        dataset = pd.read_excel(dataset_name)

    dataset = dataset.fillna(0)

    # Normalizing [Cultivated area (hectares)] feature
    if ("Cultivated area (hec)" in dataset.columns):
        for column in dataset.columns:
            if (column not in constants.excluded_normalization_columns):
                dataset[column] = dataset[column] / \
                    dataset["Cultivated area (hec)"]

    else:
        for column in dataset.columns:
            if (column not in constants.excluded_normalization_columns):
                dataset[column] = dataset[column] / dataset[0]

    # Constants are extracted manually from the dataset
    independent_variables = constants.independant_variables_potato if experiment_type == "potato" else constants.independant_variables_other_crops if experiment_type == "other" else constants.independant_variables_general
    dependent_variable = constants.dependant_variable_potato if experiment_type == "potato" else constants.dependant_variable_other_crops if experiment_type == "other" else constants.dependant_variable_general
    cft_dependant_variable = constants.cft_dependant_variable_potato if experiment_type == "potato" else constants.cft_dependant_variable_other_crops if experiment_type == "other" else constants.cft_dependant_variable_general
    
    #Applying one-hot encoder to categorical data
    if "Irrigation type" in dataset.columns:
        # Index of the column to be encoded
        index = dataset.columns.get_loc("Irrigation type")
        
        # Apply the ColumnTransformer to one-hot encode the "Irrigation type" column
        encoder = ColumnTransformer(
            [("one_hot1", OneHotEncoder(), ['Irrigation type'])],
            remainder='passthrough'
        )
        
        # Fit and transform the dataset
        dataset_transformed = encoder.fit_transform(dataset)
        
        # Create new column names for the one-hot encoded columns
        one_hot_columns = encoder.named_transformers_['one_hot1'].get_feature_names_out(['Irrigation type'])
        
        # Convert the transformed data back into a DataFrame
        dataset_transformed_df = pd.DataFrame(dataset_transformed, columns=[*dataset.columns[:index], *one_hot_columns, *dataset.columns[index+1:]])
        
        # Reset the column order based on the original DataFrame order
        dataset_transformed_df = dataset_transformed_df[dataset.columns[:index].tolist() + list(one_hot_columns) + dataset.columns[index+1:].tolist()]

        dependent_variable += 1
        cft_dependant_variable += 1
        independent_variables = range(independent_variables.start, independent_variables.stop + 1, independent_variables.step)
    else:
        dataset_transformed_df = dataset
        
    
    X = dataset_transformed_df.iloc[:, independent_variables].values
    y = dataset_transformed_df.iloc[:, dependent_variable].values if cft_evaluate == False else dataset_transformed_df.iloc[:, cft_dependant_variable].values
    
    # Scaling input and output variables features
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    X_scaled = scaler.fit_transform(X)
    y = pd.DataFrame(y_scaled)
    X = pd.DataFrame(X_scaled)
    return (dataset_transformed_df, X, y)
