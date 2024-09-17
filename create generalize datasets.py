import numpy as np
import pandas as pd
import constants


def load_dataset_from_files(files_list):
    items = np.empty([1, 1])

    for dataset_name in files_list:
        if (dataset_name.endswith("csv")):
            loaded_dataset = pd.read_csv(dataset_name)
        elif (dataset_name.endswith("xlsx")):
            loaded_dataset = pd.read_excel(dataset_name)

        if (items.shape[0] == 1):
            items = loaded_dataset.values
        else:
            items = np.concatenate((items, loaded_dataset.values))

    return items


other_crops_train_files = []
other_crops_test_files = []

for dataset in constants.experiments[-2]['datasets']:
    if (dataset['use_in_train']):
        other_crops_train_files.append(dataset['train_file_path'])

    if (dataset['use_in_test']):
        other_crops_test_files.append(dataset['test_file_path'])

potato_train_files = []
potato_test_files = []

for dataset in constants.experiments[3]['datasets']:
    if (dataset['use_in_train']):
        potato_train_files.append(dataset['train_file_path'])

    if (dataset['use_in_test']):
        potato_test_files.append(dataset['test_file_path'])

for dataset in constants.experiments[3]['aug_datasets']:
    if (dataset['use_in_train']):
        potato_train_files.append(dataset['train_file_path'])

    if (dataset['use_in_test']):
        potato_test_files.append(dataset['test_file_path'])

train_items_other_crops = load_dataset_from_files(other_crops_train_files)
test_items_other_crops = load_dataset_from_files(other_crops_test_files)
train_items_potato = load_dataset_from_files(potato_train_files)
test_items_potato = load_dataset_from_files(potato_test_files)

other_crops_columns = [0, 3, 5, 6, 7, 9, 11, 15, 16, 19, 20, 21, 22]
potato_columns = [0, 5, 1, 2, 3, 4, 10, 14, 7, 11, 15, 18, 20]

X_train = train_items_other_crops[:, other_crops_columns]
X_test = test_items_other_crops[:, other_crops_columns]

X_train = np.concatenate((X_train, train_items_potato[:, potato_columns]))
X_test = np.concatenate((X_test, test_items_potato[:, potato_columns]))
  
output_dataset_columns = ['Cultivated area (hectares)', 'Seed (Kg)', 'Urea (kg)', 'Phosphate (kg)',
                          'Potassium (kg)', 'Nitrogen (kg)', 'Insecticide (lit)', 'Water (m3)',
                          'Irrigation frequency', 'Yield (kg)', 'Fuel (lit)', 'Electricity (kwh)',
                          'Calculated carbon (kg co2-eq)']

train_df = pd.DataFrame(X_train, columns = output_dataset_columns)
train_df.to_csv("data/generalized-all-crops - Train.csv")

test_df = pd.DataFrame(X_test, columns = output_dataset_columns)
test_df.to_csv("data/generalized-all-crops - Test.csv")