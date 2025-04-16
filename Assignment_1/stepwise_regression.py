import numpy as np
from dataloader import select_features, add_intercept
from regression_utils import (
    gradient_descent,
    predict,
    bic,
    r_squared,
    adjusted_r_squared
)


def evaluate_model_on_train_test(
    train_data,
    test_data,
    feature_list,
    target_name,
    theta
):
    """
    Evaluates a fitted model (theta) on both the train and test datasets
    using the specified features and target.
    """
    # Evaluate on train like in tasks 1 & 2
    X_train, y_train = select_features(train_data, feature_list, target_name)
    X_train = add_intercept(X_train)
    y_pred_train = predict(X_train, theta)  # use theta for predictions on train

    # Calculate mse, r^2, and adj r^2 for train
    mse_train = np.mean((y_pred_train - y_train)**2)
    r2_train = r_squared(y_train, y_pred_train)
    adj_r2_train = adjusted_r_squared(y_train, y_pred_train, num_features=len(feature_list))

    # Evaluate on test like in tasks 1 & 2
    X_test, y_test = select_features(test_data, feature_list, target_name)
    X_test = add_intercept(X_test)
    y_pred_test = predict(X_test, theta)  # use theta for predictions on test

    # Calculate mse, r^2, and adj r^2 for test
    mse_test = np.mean((y_pred_test - y_test)**2)
    r2_test = r_squared(y_test, y_pred_test)
    adj_r2_test = adjusted_r_squared(y_test, y_pred_test, num_features=len(feature_list))

    # Store metrics as dict
    metrics = {
        "train_mse": mse_train,
        "test_mse": mse_test,
        "train_r2": r2_train,
        "test_r2": r2_test,
        "train_adj_r2": adj_r2_train,
        "test_adj_r2": adj_r2_test
    }
    return metrics


def train_and_evaluate(
        train_data,
        feature_list,
        target_name,
        alpha=0.01,
        num_iterations=1000,
        metric='bic'
):
    """
    Helper function to build X and y for the given feature_list from train_data,
    add the intercept to X, performs gradient descent to obtain theta, to
    predict on X and compute the chosen metric (BIC, R², or Adjusted R²).
    """
    # Build X, y
    X, y = select_features(train_data, feature_list, target_name)

    # Add intercept
    X = add_intercept(X)

    # Fit model using gradient descent
    theta, cost_history = gradient_descent(X, y, alpha=alpha, num_iterations=num_iterations)

    # Evaluate predictions
    y_pred = predict(X, theta)

    # Compute chosen metric
    if metric.lower() == 'bic':
        # BIC -> lower is better
        metric_value = bic(y, y_pred, num_features=len(feature_list))
    elif metric.lower() == 'r2':
        # R^2 -> higher is better
        metric_value = r_squared(y, y_pred)
    elif metric.lower() == 'adjusted_r2':
        # Adjusted R^2 -> higher is better
        metric_value = adjusted_r_squared(y, y_pred, num_features=len(feature_list))
    else:
        raise ValueError("Unknown metric. Supported: 'bic', 'r2', 'adjusted_r2'")

    return metric_value, theta


def forward_stepwise(
        train_data,
        feature_candidates,
        target_name,
        max_features=5,
        alpha=0.01,
        num_iterations=1000,
        criterion='bic'
):
    """
    Implements forward stepwise regression on the training data by starting
    with an empty list of selected features, iteratively adding one feature
    at a time from 'feature_candidates' that best improves the model based
    on the chosen criterion, stop when 'max_features' is reached or when no
    improvement is possible.
    """
    # Initialize lists for features
    selected_features = []
    remaining_features = feature_candidates[:]
    history = []

    # For 'bic' -> minimize, for 'r2'/'adjusted_r2' -> maximize
    if criterion.lower() == 'bic':
        best_metric_so_far = float('inf')
        is_lower_better = True
    else:
        best_metric_so_far = float('-inf')
        is_lower_better = False

    iteration = 0

    # Loop until max features is reached or no features are remaining
    while len(selected_features) < max_features and len(remaining_features) > 0:
        iteration += 1
        feature_to_add = None
        best_metric_this_round = None

        # Try to add each candidate feature
        for f in remaining_features:
            trial_features = selected_features + [f]
            metric_value, _ = train_and_evaluate(
                train_data,
                trial_features,
                target_name,
                alpha=alpha,
                num_iterations=num_iterations,
                metric=criterion
            )

            # Decide to add feature depending on evaluation criterion
            if is_lower_better:
                if metric_value < (best_metric_this_round if best_metric_this_round is not None else float('inf')):
                    best_metric_this_round = metric_value
                    feature_to_add = f
            else:
                if metric_value > (best_metric_this_round if best_metric_this_round is not None else float('-inf')):
                    best_metric_this_round = metric_value
                    feature_to_add = f

        # Decide whether to add the best candidate
        if feature_to_add is not None:
            # Check if there's improvement
            if is_lower_better and best_metric_this_round < best_metric_so_far:
                selected_features.append(feature_to_add)
                remaining_features.remove(feature_to_add)
                best_metric_so_far = best_metric_this_round

                # Retrain with updated list
                _, best_theta = train_and_evaluate(
                    train_data,
                    selected_features,
                    target_name,
                    alpha=alpha,
                    num_iterations=num_iterations,
                    metric=criterion
                )
                # Append added feature to history list
                history.append({
                    'iteration': iteration,
                    'chosen_feature': feature_to_add,
                    'metric_value': best_metric_so_far,
                    'features_set': selected_features[:]
                })
            elif (not is_lower_better) and best_metric_this_round > best_metric_so_far:
                selected_features.append(feature_to_add)
                remaining_features.remove(feature_to_add)
                best_metric_so_far = best_metric_this_round

                # Retrain with updated list
                _, best_theta = train_and_evaluate(
                    train_data,
                    selected_features,
                    target_name,
                    alpha=alpha,
                    num_iterations=num_iterations,
                    metric=criterion
                )
                # Append added feature to history list
                history.append({
                    'iteration': iteration,
                    'chosen_feature': feature_to_add,
                    'metric_value': best_metric_so_far,
                    'features_set': selected_features[:]
                })
            else:
                # No improvement -> break
                break
        else:
            break

    # Final check if best_theta wasn't set
    if 'best_theta' not in locals():
        # This is just in case no features were added
        _, best_theta = train_and_evaluate(
            train_data,
            selected_features,
            target_name,
            alpha=alpha,
            num_iterations=num_iterations,
            metric=criterion
        )

    return selected_features, best_theta, history


def backward_stepwise(
        train_data,
        feature_candidates,
        target_name,
        num_features_to_remove=5,
        alpha=0.01,
        num_iterations=1000,
        criterion='bic'
):
    """
    Implements backward stepwise regression on the training data by starting with
    the full set of 'feature_candidates', iteratively removing one feature at a
    time if it improves the chosen metric, stopping after removing set number of
    features or if there was no improvement.
    """

    # Initialize lists for features
    selected_features = feature_candidates[:]
    history = []

    # Train initial model
    initial_metric, _ = train_and_evaluate(
        train_data,
        selected_features,
        target_name,
        alpha=alpha,
        num_iterations=num_iterations,
        metric=criterion
    )

    # Set best using criterion selected
    if criterion.lower() == 'bic':
        best_metric_so_far = initial_metric
        is_lower_better = True
    else:
        best_metric_so_far = initial_metric
        is_lower_better = False

    iteration = 0

    # Loop while there are still selected features and less than number of features to remove
    while len(selected_features) > 0 and iteration < num_features_to_remove:
        iteration += 1
        feature_to_remove = None
        best_metric_this_round = None

        # Remove each feature
        for f in selected_features:
            trial_features = [feat for feat in selected_features if feat != f]
            metric_value, _ = train_and_evaluate(
                train_data,
                trial_features,
                target_name,
                alpha=alpha,
                num_iterations=num_iterations,
                metric=criterion
            )

            if is_lower_better:
                if (best_metric_this_round is None) or (metric_value < best_metric_this_round):
                    best_metric_this_round = metric_value
                    feature_to_remove = f
            else:
                if (best_metric_this_round is None) or (metric_value > best_metric_this_round):
                    best_metric_this_round = metric_value
                    feature_to_remove = f

        # Calculate if improvement
        improvement = (
            (best_metric_this_round < best_metric_so_far)
            if is_lower_better
            else (best_metric_this_round > best_metric_so_far)
        )

        # If improved, remove the feature
        if improvement:
            selected_features.remove(feature_to_remove)
            best_metric_so_far = best_metric_this_round

            # Retrain with updated list
            _, best_theta = train_and_evaluate(
                train_data,
                selected_features,
                target_name,
                alpha=alpha,
                num_iterations=num_iterations,
                metric=criterion
            )
            # Append removed feature to history list
            history.append({
                'iteration': iteration,
                'removed_feature': feature_to_remove,
                'metric_value': best_metric_so_far,
                'features_set': selected_features[:]
            })
        else:
            break

    if 'best_theta' not in locals():
        # Retrain model if best theta not in local variable lists
        _, best_theta = train_and_evaluate(
            train_data,
            selected_features,
            target_name,
            alpha=alpha,
            num_iterations=num_iterations,
            metric=criterion
        )

    return selected_features, best_theta, history
