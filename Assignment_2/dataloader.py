import csv
import random
import numpy as np


def load_data(filepath):
    """
    Loads the data from CSV into a dictionary of columns.
    Each key will be the column name, and each value is a list of floats or strings.
    """
    data_dict = {}

    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header row

        # Initialize dictionary keys
        for col_name in header:
            data_dict[col_name] = []

        for row in reader:
            # Skip empty lines, if any
            if not row:
                continue

            # Append each column value to the corresponding list
            for col_name, value in zip(header, row):
                # If the column is "id" or "outcome" which could be a string (N or Y),
                # skip converting to float. Otherwise, attempt to convert to float.
                if col_name in ["id", "outcome"]:
                    data_dict[col_name].append(value)
                else:
                    try:
                        data_dict[col_name].append(float(value))
                    except ValueError:
                        data_dict[col_name].append(None)

    return data_dict


def remove_rows_with_none(data_dict, columns_of_interest):
    """
    Removes any row where any of the specified columns has None.
    Returns a new dictionary with those rows dropped.
    """
    clean_data = {col: [] for col in data_dict}

    # Determine how many rows are in the data by checking length of any column
    n = len(next(iter(data_dict.values())))

    for i in range(n):
        keep_row = True
        for col in columns_of_interest:
            if data_dict[col][i] is None:
                keep_row = False
                break
        if keep_row:
            for col in data_dict:
                clean_data[col].append(data_dict[col][i])

    return clean_data


def train_test_split(data_dict, test_size=0.2, shuffle=True, random_state=None):
    """
    Splits the data (column-wise dict) into training and testing sets.
    If `shuffle` is True, data is shuffled before splitting.
    If `random_state` is provided, it seeds the random number generator for reproducible shuffling.
    """
    any_column = next(iter(data_dict))
    n = len(data_dict[any_column])  # total number of rows

    indices = list(range(n))
    if shuffle:
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(indices)

    split_index = int(n * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_dict = {}
    test_dict = {}

    for col_name, col_values in data_dict.items():
        train_dict[col_name] = [col_values[i] for i in train_indices]
        test_dict[col_name] = [col_values[i] for i in test_indices]

    return train_dict, test_dict


import random


def stratified_train_test_split(data_dict, test_size=0.2, random_state=None):
    """
    Splits data_dict into train and test sets, preserving the approximate
    ratio of 0 vs. 1 in 'outcome'. Returns (train_dict, test_dict).

    - data_dict is a column-wise dict, including a key "outcome" with 0/1 labels.
    - test_size is the fraction that goes into test.
    - random_state is an optional seed for reproducibility.
    """
    if random_state is not None:
        random.seed(random_state)

    # 1) Separate row indices by class label
    indices_0 = []
    indices_1 = []
    any_col = next(iter(data_dict))
    n = len(data_dict[any_col])  # total rows
    for i in range(n):
        if data_dict["outcome"][i] == 1:
            indices_1.append(i)
        else:
            indices_0.append(i)

    # 2) Shuffle each list (to randomize which rows go train/test)
    random.shuffle(indices_0)
    random.shuffle(indices_1)

    # 3) Determine how many go into test for each class
    test_count_0 = int(len(indices_0) * test_size)
    test_count_1 = int(len(indices_1) * test_size)

    # 4) Slice out test vs. train for each class
    test_indices_0 = indices_0[:test_count_0]
    test_indices_1 = indices_1[:test_count_1]
    train_indices_0 = indices_0[test_count_0:]
    train_indices_1 = indices_1[test_count_1:]

    # 5) Combine them
    test_indices = test_indices_0 + test_indices_1
    train_indices = train_indices_0 + train_indices_1

    # 6) Shuffle again if desired
    random.shuffle(test_indices)
    random.shuffle(train_indices)

    # 7) Rebuild train_dict, test_dict
    train_dict = {}
    test_dict = {}
    for col_name, col_values in data_dict.items():
        train_dict[col_name] = [col_values[i] for i in train_indices]
        test_dict[col_name] = [col_values[i] for i in test_indices]

    return train_dict, test_dict


def select_features(data_dict, feature_names, target_name):
    """
    Given a column-wise data dictionary, returns an (X, y) pair for modeling.
    X has shape (m, len(feature_names)), y has shape (m,).
    """
    n = len(data_dict[target_name])  # number of samples
    X = []
    for i in range(n):
        row_features = []
        for feat in feature_names:
            row_features.append(data_dict[feat][i])
        X.append(row_features)
    X = np.array(X, dtype=float)

    y = np.array(data_dict[target_name], dtype=float)
    return X, y


def add_intercept(X):
    """
    Adds a column of 1s (intercept) to the left of X.
    """
    n_samples = X.shape[0]
    intercept_col = np.ones((n_samples, 1), dtype=float)
    X_with_intercept = np.hstack((intercept_col, X))
    return X_with_intercept


def feature_scaling(X, method="standardize"):
    """
    Scales the features in X based on the specified method.
    Returns scaled X and a dict of parameters used for scaling.
    """
    if method == "standardize":
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0, ddof=1)  # sample std
        # avoid divide-by-zero
        stds_no_zero = np.where(stds == 0, 1, stds)
        X_scaled = (X - means) / stds_no_zero
        params = {
            "method": "standardize",
            "means": means,
            "stds": stds
        }
        return X_scaled, params
    elif method == "minmax":
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        range_vals = np.where((max_vals - min_vals)==0, 1, (max_vals - min_vals))
        X_scaled = (X - min_vals) / range_vals
        params = {
            "method": "minmax",
            "min_vals": min_vals,
            "max_vals": max_vals
        }
        return X_scaled, params
    else:
        return X, {}
