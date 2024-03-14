import matplotlib.pyplot as plt

print_training_logs = 0

independant_variables_other_crops = range(1, 22)
dependant_variable_other_crops = 22
cft_dependant_variable_other_crops = 23

independant_variables_potato = range(1, 19)
dependant_variable_potato = 19
cft_dependant_variable_potato = 20

train_test_datasets_file_name = "train_test_datasets.sav"
train_test_datasets_aug_file_name = "train_test_datasets_aug.sav"
scores_output_file_name = "scores.txt"
scores_output_csv_file = "result.csv"
scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_median_absolute_error', 'neg_mean_squared_log_error'] #strategy to evaluate the performance of the cross-validation model

excluded_normalization_columns = [
    "Mixed tillage (Frequency)",
    "Other operations(Frequency)",
    "Seeding(Frequency)",
    "Fertilizing(Frequency)",
    "spraying(Frequency)",
    "cultivator(Frequency)",
    "Watering(Frequency)",
    "Harvesting(Frequency)",
    "Irrigation type"]

experiments = [
    # 1. Iranian potato only
    # 2. Iranian potato only - aug
    # 3. Iranian potato only - Cool farm tool,
    # 4. Iranian potato only - Cool farm tool - aug
    {
        "title": "1. Iranian potato only",
        "experiment_type": "potato",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/October datasets/Iran-potato - Train.xlsx',
                "test_file_path": 'data/October datasets/Iran-potato - Test.xlsx',
                "dataset_displayname": 'Iran potato',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": [
            {
                "train_file_path": 'data/October datasets/IranPotato_Aug - Train.csv',
                "test_file_path": 'data/October datasets/IranPotato_Aug - Test.csv',
                "dataset_displayname": 'Iran potato aug',
                "use_in_train": True,
                "use_in_test": False
            }
        ]
    },
    
    # 5. Morocco potato only
    # 6. Morocco potato only - aug
    # 7. Morocco potato only - Cool farm tool,
    # 8. Morocco potato only - Cool farm tool - aug
    {
        "title": "5. Morocco potato only",
        "experiment_type": "potato",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/October datasets/Morocco-potato - Train.xlsx',
                "test_file_path": 'data/October datasets/Morocco-potato - Test.xlsx',
                "dataset_displayname": 'Morocco potato',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": [
            {
                "train_file_path": 'data/October datasets/MoroccoPotato_Aug - Train.csv',
                "test_file_path": 'data/October datasets/MoroccoPotato_Aug - Test.csv',
                "dataset_displayname": 'Morocco potato aug',
                "use_in_train": True,
                "use_in_test": False
            }
        ]
    },
    
    # 9.  Iranian Train set only(0.8) + Iran(0.2) & Mor test set(0.5) & Peru data(1)
    # 10. Iranian Train set only(0.8) + Iran(0.2) & Mor test set(0.5) & Peru data(1) - aug
    # 11. Iranian Train set only(0.8) + Iran(0.2) & Mor test set(0.5) & Peru data(1) - Cool farm tool
    # 12. Iranian Train set only(0.8) + Iran(0.2) & Mor test set(0.5) & Peru data(1) - Cool farm tool - aug
    {
        "title": "9. Iranian Train set only(0.8) + Iran(0.2) & Mor test set(0.5) & Peru data(1)",
        "experiment_type": "potato",
        "generate_cft_result": False,
        "datasets": [
            {
                "train_file_path": 'data/October datasets/Iran-potato - Train.xlsx',
                "test_file_path": 'data/October datasets/Iran-potato - Test.xlsx',
                "dataset_displayname": 'Iran potato',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/October datasets/Morocco-potato - Train.xlsx',
                "test_file_path": 'data/October datasets/Morocco-potato - Test.xlsx',
                "dataset_displayname": 'Morocco potato',
                "use_in_train": False,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/October datasets/Peru-potato - Test.xlsx',
                "test_file_path": 'data/October datasets/Peru-potato - Test.xlsx',
                "dataset_displayname": 'Peru potato',
                "use_in_train": False,
                "use_in_test": True
            },
        ],
        "aug_datasets": [
            {
                "train_file_path": 'data/October datasets/IranPotato_Aug - Train.csv',
                "test_file_path": 'data/October datasets/IranPotato_Aug - Test.csv',
                "dataset_displayname": 'Iran potato aug',
                "use_in_train": True,
                "use_in_test": False
            },
            {
                "train_file_path": 'data/October datasets/MoroccoPotato_Aug - Train.csv',
                "test_file_path": 'data/October datasets/MoroccoPotato_Aug - Test.csv',
                "dataset_displayname": 'Morocco potato aug',
                "use_in_train": True,
                "use_in_test": False
            }
        ]
    },
    
    # 13. Iranian Train set (0.8) & Mor Train set (0.5) + Iran(0.2) & Mor test set(0.5) & Peru data(1)
    # 14. Iranian Train set (0.8) & Mor Train set (0.5) + Iran(0.2) & Mor test set(0.5) & Peru data(1) - aug
    # 15. Iranian Train set (0.8) & Mor Train set (0.5) + Iran(0.2) & Mor test set(0.5) & Peru data(1) - Cool farm tool
    # 16. Iranian Train set (0.8) & Mor Train set (0.5) + Iran(0.2) & Mor test set(0.5) & Peru data(1) - Cool farm tool - aug
    {
        "title": "13. Iranian Train set (0.8) & Mor Train set (0.5) + Iran(0.2) & Mor test set(0.5) & Peru data(1)",
        "experiment_type": "potato",
        "generate_cft_result": False,
        "datasets": [
            {
                "train_file_path": 'data/October datasets/Iran-potato - Train.xlsx',
                "test_file_path": 'data/October datasets/Iran-potato - Test.xlsx',
                "dataset_displayname": 'Iran potato',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/October datasets/Morocco-potato - Train.xlsx',
                "test_file_path": 'data/October datasets/Morocco-potato - Test.xlsx',
                "dataset_displayname": 'Morocco potato',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/October datasets/Peru-potato - Test.xlsx',
                "test_file_path": 'data/October datasets/Peru-potato - Test.xlsx',
                "dataset_displayname": 'Peru potato',
                "use_in_train": False,
                "use_in_test": True
            },
        ],
        "aug_datasets": [
            {
                "train_file_path": 'data/October datasets/IranPotato_Aug - Train.csv',
                "test_file_path": 'data/October datasets/IranPotato_Aug - Test.csv',
                "dataset_displayname": 'Iran potato aug',
                "use_in_train": True,
                "use_in_test": False
            },
            {
                "train_file_path": 'data/October datasets/MoroccoPotato_Aug - Train.csv',
                "test_file_path": 'data/October datasets/MoroccoPotato_Aug - Test.csv',
                "dataset_displayname": 'Morocco potato aug',
                "use_in_train": True,
                "use_in_test": False
            }
        ]
    },
    
    # 17. Cabbage only
    {
        "title": "17. Cabbage only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Vegetables/Cabbage_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Cabbage_aug - Final - Test.csv',
                "dataset_displayname": 'Cabbage',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 18. Carrot only
    {
        "title": "18. Carrot only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Vegetables/Carrot_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Carrot_aug - Final - Test.csv',
                "dataset_displayname": 'Carrot',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 19. Cucumber only
    {
        "title": "19. Cucumber only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Vegetables/Cucumber_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Cucumber_aug - Final - Test.csv',
                "dataset_displayname": 'Cucumber',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 20. Onion only
    {
        "title": "20. Onion only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Vegetables/Onion_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Onion_aug - Final - Test.csv',
                "dataset_displayname": 'Onion',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 21. Sugar beet only
    {
        "title": "21. Sugar beet only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Vegetables/Sugar beet_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Sugar beet_aug - Final - Test.csv',
                "dataset_displayname": 'Sugar beet',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 22. Tomato only
    {
        "title": "22. Tomato only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Vegetables/Tomato_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Tomato_aug - Final  - Test.csv',
                "dataset_displayname": 'Tomato',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 23. Watermelon only
    {
        "title": "23. Watermelon only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Vegetables/Watermelon_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Watermelon_aug - Final - Test.csv',
                "dataset_displayname": 'Watermelon',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 24. Alfalfa only
    {
        "title": "24. Alfalfa only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [
            {
                "train_file_path": 'data/Lentils/Alfalfa_aug - Final - Train.csv',
                "test_file_path": 'data/Lentils/Alfalfa_aug - Final - Test.csv',
                "dataset_displayname": 'Alfalfa',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 25. Beans only
    {
        "title": "25. Beans only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [
            {
                "train_file_path": 'data/Lentils/Beans_aug - Final - Train.csv',
                "test_file_path": 'data/Lentils/Beans_aug - Final - Test.csv',
                "dataset_displayname": 'Beans',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 26. Lentils only
    {
        "title": "26. Lentils only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [
            {
                "train_file_path": 'data/Lentils/lentils2_aug - Final - Train.csv',
                "test_file_path": 'data/Lentils/lentils2_aug - Final - Test.csv',
                "dataset_displayname": 'Lentils',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 27. Irrigated wheat only
    {
        "title": "27. Irrigated wheat only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [
            {
                "train_file_path": 'data/Cereals/Irrigated wheat_aug - Final - Train.csv',
                "test_file_path": 'data/Cereals/Irrigated wheat_aug - Final - Test.csv',
                "dataset_displayname": 'Irrigated wheat',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 28. Corn only
    {
        "title": "28. Corn only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Cereals/Corn_aug - Final - Train.csv',
                "test_file_path": 'data/Cereals/Corn_aug - Final - Test.csv',
                "dataset_displayname": 'Corn',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 29. Irrigated barley only
    {
        "title": "29. Irrigated barley only",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Cereals/Irrigated barley_aug - Final - Train.csv',
                "test_file_path": 'data/Cereals/Irrigated barley_aug - Final - Test.csv',
                "dataset_displayname": 'Irrigated barley',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 31. Cereals together
    {
        "title": "31. Cereals together",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [
            {
                "train_file_path": 'data/Cereals/Irrigated wheat_aug - Final - Train.csv',
                "test_file_path": 'data/Cereals/Irrigated wheat_aug - Final - Test.csv',
                "dataset_displayname": 'Irrigated wheat',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Cereals/Corn_aug - Final - Train.csv',
                "test_file_path": 'data/Cereals/Corn_aug - Final - Test.csv',
                "dataset_displayname": 'Corn',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Cereals/Irrigated barley_aug - Final - Train.csv',
                "test_file_path": 'data/Cereals/Irrigated barley_aug - Final - Test.csv',
                "dataset_displayname": 'Irrigated barley',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 32. Lentils together
    {
        "title": "32. Lentils together",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Lentils/Alfalfa_aug - Final - Train.csv',
                "test_file_path": 'data/Lentils/Alfalfa_aug - Final - Test.csv',
                "dataset_displayname": 'Alfalfa',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Lentils/Beans_aug - Final - Train.csv',
                "test_file_path": 'data/Lentils/Beans_aug - Final - Test.csv',
                "dataset_displayname": 'Beans',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Lentils/lentils2_aug - Final - Train.csv',
                "test_file_path": 'data/Lentils/lentils2_aug - Final - Test.csv',
                "dataset_displayname": 'Lentils',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
    
    # 33. Vegetables together
    {
        "title": "33. Vegetables together",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Vegetables/Cabbage_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Cabbage_aug - Final - Test.csv',
                "dataset_displayname": 'Cabbage',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Carrot_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Carrot_aug - Final - Test.csv',
                "dataset_displayname": 'Carrot',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Cucumber_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Cucumber_aug - Final - Test.csv',
                "dataset_displayname": 'Cucumber',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Onion_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Onion_aug - Final - Test.csv',
                "dataset_displayname": 'Onion',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Sugar beet_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Sugar beet_aug - Final - Test.csv',
                "dataset_displayname": 'Sugar beet',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Tomato_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Tomato_aug - Final  - Test.csv',
                "dataset_displayname": 'Tomato',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Watermelon_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Watermelon_aug - Final - Test.csv',
                "dataset_displayname": 'Watermelon',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },

    # 30. All 13 crops of Iran together
    {
        "title": "30. All 13 crops of Iran together",
        "experiment_type": "other",
        "generate_cft_result": False,
        "datasets": [ 
            {
                "train_file_path": 'data/Vegetables/Cabbage_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Cabbage_aug - Final - Test.csv',
                "dataset_displayname": 'Cabbage',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Carrot_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Carrot_aug - Final - Test.csv',
                "dataset_displayname": 'Carrot',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Cucumber_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Cucumber_aug - Final - Test.csv',
                "dataset_displayname": 'Cucumber',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Onion_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Onion_aug - Final - Test.csv',
                "dataset_displayname": 'Onion',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Sugar beet_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Sugar beet_aug - Final - Test.csv',
                "dataset_displayname": 'Sugar beet',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Tomato_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Tomato_aug - Final  - Test.csv',
                "dataset_displayname": 'Tomato',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Vegetables/Watermelon_aug - Final - Train.csv',
                "test_file_path": 'data/Vegetables/Watermelon_aug - Final - Test.csv',
                "dataset_displayname": 'Watermelon',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Lentils/Alfalfa_aug - Final - Train.csv',
                "test_file_path": 'data/Lentils/Alfalfa_aug - Final - Test.csv',
                "dataset_displayname": 'Alfalfa',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Lentils/Beans_aug - Final - Train.csv',
                "test_file_path": 'data/Lentils/Beans_aug - Final - Test.csv',
                "dataset_displayname": 'Beans',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Lentils/lentils2_aug - Final - Train.csv',
                "test_file_path": 'data/Lentils/lentils2_aug - Final - Test.csv',
                "dataset_displayname": 'Lentils',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Cereals/Irrigated wheat_aug - Final - Train.csv',
                "test_file_path": 'data/Cereals/Irrigated wheat_aug - Final - Test.csv',
                "dataset_displayname": 'Irrigated wheat',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Cereals/Corn_aug - Final - Train.csv',
                "test_file_path": 'data/Cereals/Corn_aug - Final - Test.csv',
                "dataset_displayname": 'Corn',
                "use_in_train": True,
                "use_in_test": True
            },
            {
                "train_file_path": 'data/Cereals/Irrigated barley_aug - Final - Train.csv',
                "test_file_path": 'data/Cereals/Irrigated barley_aug - Final - Test.csv',
                "dataset_displayname": 'Irrigated barley',
                "use_in_train": True,
                "use_in_test": True
            },
        ],
        "aug_datasets": []
    },
]