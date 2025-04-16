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
                # If the column is "outcome" which could be string (N or Y), skip converting to float
                # For the other columns, try to convert numeric val to float:
                if col_name in ["id", "outcome"]:
                    # Keep them as string
                    data_dict[col_name].append(value)
                else:
                    # Attempt to convert to float
                    try:
                        data_dict[col_name].append(float(value))
                    except ValueError:
                        # If there's a problem converting just set to None
                        data_dict[col_name].append(None)

    return data_dict


def train_test_split(data_dict, test_size=0.2, shuffle=True, random_state=None):
    """
    Splits the data (column-wise dict) into training and testing sets. If `shuffle` is
    True, data is shuffled before splitting. If `random_state` is provided, it seeds
    the random number generator for reproducible shuffling.
    """
    # Find the total number of rows by looking at column length
    any_column = next(iter(data_dict))
    n = len(data_dict[any_column])  # get number of rows

    # Create a list of all row indices from 0 to n-1
    indices = list(range(n))

    # Shuffle indices if specified
    if shuffle:
        # If random_state specified, seed the random generator
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(indices)

    # Compute split point using test size
    split_index = int(n * (1 - test_size))

    # Create slices into train and test portions
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # Initialize train/test dicts to output
    train_dict = {}
    test_dict = {}

    # For each column, pick the rows appropriately from train or test indices
    for col_name, col_values in data_dict.items():
        train_dict[col_name] = [col_values[i] for i in train_indices]
        test_dict[col_name] = [col_values[i] for i in test_indices]

    return train_dict, test_dict


def select_features(data_dict, feature_names, target_name):
    """
    Given a column-wise data dictionary, returns an (X, y) pair for modeling.
    """
    # Number of samples
    n = len(data_dict[target_name])

    # Build X
    # For each row i, collect data_dict[feat][i] for feat in feature_names
    X = []
    for i in range(n):
        row_features = []
        for feat in feature_names:
            row_features.append(data_dict[feat][i])
        X.append(row_features)

    # Convert to NumPy
    X = np.array(X, dtype=float)

    # Build y
    y = np.array(data_dict[target_name], dtype=float)

    return X, y


def add_intercept(X):
    """
    Adds an intercept (a column of 1s) to the feature matrix X.
    """
    # Number of samples
    n_samples = X.shape[0]

    # Create a column of ones of shape (n_samples, 1)
    intercept_col = np.ones((n_samples, 1), dtype=float)

    # Concatenate intercept_col and X along the horizontal axis
    X_with_intercept = np.hstack((intercept_col, X))

    return X_with_intercept


def feature_scaling(X, method="standardize"):
    """
    Scales the features in X based on the specified method.
    """
    # If standardization is requested
    if method == "standardize":
        means = np.mean(X, axis=0)  # Compute mean of each column
        stds = np.std(X, axis=0, ddof=1)  # Compute sample standard deviation (ddof=1) for each column

        # Replace zero stds with 1 to avoid divide-by-zero
        stds_no_zero = np.where(stds == 0, 1, stds)

        # Subtract mean and divide by std to standardize each feature
        X_scaled = (X - means) / stds_no_zero

        # Save the parameters used for potential inverse transform / test set scaling
        params = {
            "method": "standardize",
            "means": means,
            "stds": stds
        }
        return X_scaled, params

    # Might not need to use min/max -- possibly unneeded
    elif method == "minmax":
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)

        # Compute range for each column, substituting 1 if range is zero
        range_vals = np.where((max_vals - min_vals) == 0, 1, (max_vals - min_vals))

        # Scale each feature to [0,1] using min and max
        X_scaled = (X - min_vals) / range_vals

        # Save parameters for possible later use
        params = {
            "method": "minmax",
            "min_vals": min_vals,
            "max_vals": max_vals
        }
        return X_scaled, params

    else:
        # If unknown method, return X unchanged
        return X, {}


def remove_rows_with_none(data_dict, columns_of_interest):
    """
    Removes any row where any of the specified columns has None.
    Returns a new dictionary with those rows dropped.
    """
    # Create a new dict with the same columns, but initially empty
    clean_data = {col: [] for col in data_dict}

    # Determine how many rows are in the data by checking length of any column
    n = len(list(data_dict.values())[0])

    # Iterate over each row index
    for i in range(n):
        keep_row = True
        # Check each column of interest to see if its value is None in this row
        for col in columns_of_interest:
            if data_dict[col][i] is None:
                keep_row = False
                break  # No need to check other columns if one is None
        # If row is valid (no Nones in columns of interest), copy it to clean_data
        if keep_row:
            # Append row i's values to clean_data
            for col in data_dict:
                clean_data[col].append(data_dict[col][i])

    return clean_data
