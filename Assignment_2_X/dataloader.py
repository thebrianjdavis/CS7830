import csv
import random
import numpy as np


def load_data(filepath):
    """
    Loads data from a CSV file into a dictionary of columns.
    Each key is a column name, and each value is a list of floats or strings.
    """
    data_dict = {}
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Initialize data_dict keys
        for col in header:
            data_dict[col] = []

        for row in reader:
            # Skip empty lines
            if not row:
                continue

            for col_name, value in zip(header, row):
                if col_name == "id" or col_name == "outcome":
                    # keep 'id' and 'outcome' as strings (or outcome to be encoded later)
                    data_dict[col_name].append(value)
                else:
                    # attempt to convert numeric columns to float
                    try:
                        data_dict[col_name].append(float(value))
                    except ValueError:
                        data_dict[col_name].append(None)

    return data_dict


def train_test_split(data_dict, test_size=0.2, shuffle=True, random_state=None):
    """
    Splits data into training and testing sets (column-wise dict).
    If shuffle=True, the rows are randomized before splitting.
    """
    any_col = next(iter(data_dict))
    n = len(data_dict[any_col])
    indices = list(range(n))

    # Shuffle
    if shuffle:
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(indices)

    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_dict = {}
    test_dict = {}

    for col in data_dict:
        train_dict[col] = [data_dict[col][i] for i in train_idx]
        test_dict[col] = [data_dict[col][i] for i in test_idx]

    return train_dict, test_dict


def select_features(data_dict, feature_names, target_name):
    """
    Given a column-wise data dictionary, returns an (X, y) pair.
    Skips rows that contain None in the specified features or target.
    """
    X_list = []
    y_list = []
    n = len(data_dict[target_name])

    for i in range(n):
        row_feats = []
        row_ok = True
        for feat in feature_names:
            val = data_dict[feat][i]
            if val is None:
                row_ok = False
                break
            row_feats.append(val)
        target_val = data_dict[target_name][i]
        if target_val is None:
            row_ok = False

        if row_ok:
            X_list.append(row_feats)
            y_list.append(target_val)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y


def add_intercept(X):
    """
    Adds a column of 1's as intercept to X.
    """
    m = X.shape[0]
    intercept_col = np.ones((m, 1))
    return np.hstack((intercept_col, X))


def remove_rows_with_none(data_dict, columns_of_interest):
    """
    Removes rows from data_dict if any column in columns_of_interest is None.
    Returns a new dictionary with those rows dropped.
    """
    clean_data = {col: [] for col in data_dict}
    n = len(list(data_dict.values())[0])

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


def feature_scaling(X):
    """
    Standardizes columns in X using z-score:
      X_scaled = (X - mean) / std
    Returns X_scaled and a dict with the means and stds for each column.
    """
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)

    # Avoid divide-by-zero
    stds_no_zero = np.where(stds == 0, 1, stds)

    X_scaled = (X - means) / stds_no_zero

    params = {
        "means": means,
        "stds": stds
    }
    return X_scaled, params
