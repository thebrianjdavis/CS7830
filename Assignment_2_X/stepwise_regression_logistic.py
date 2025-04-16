import numpy as np
from logistic_regression_utils import (
    logistic_gradient_descent,
    logistic_cost_function,
    logistic_predict_class
)
from dataloader import select_features, add_intercept


def forward_selection_logistic(
    train_data,
    target_name,
    feature_candidates,
    alpha=0.01,
    num_iterations=1000,
    max_features=None
):
    """
    Implements forward selection for logistic regression.
    Starting with no features, iteratively add the feature
    that yields the lowest logistic cost (or best improvement).
    """
    if max_features is None:
        max_features = len(feature_candidates)

    selected_features = []
    remaining = feature_candidates[:]
    best_cost_so_far = float('inf')
    history = []

    while len(selected_features) < max_features and len(remaining) > 0:
        feature_to_add = None
        best_local_cost = float('inf')

        for f in remaining:
            trial_features = selected_features + [f]
            X_train, y_train = select_features(train_data, trial_features, target_name)
            X_train = add_intercept(X_train)

            # Fit logistic regression
            theta_tmp, cost_hist_tmp = logistic_gradient_descent(
                X_train, y_train, alpha=alpha, num_iterations=num_iterations
            )
            cost_val = logistic_cost_function(X_train, y_train, theta_tmp)
            if cost_val < best_local_cost:
                best_local_cost = cost_val
                feature_to_add = f

        # Check if actually improved cost
        if feature_to_add is not None and best_local_cost < best_cost_so_far:
            selected_features.append(feature_to_add)
            remaining.remove(feature_to_add)
            best_cost_so_far = best_local_cost

            # Retrain to get best_theta
            X_final, y_final = select_features(train_data, selected_features, target_name)
            X_final = add_intercept(X_final)
            best_theta, _ = logistic_gradient_descent(X_final, y_final, alpha=alpha, num_iterations=num_iterations)

            history.append({
                "selected_features": selected_features[:],
                "cost_value": best_cost_so_far
            })
        else:
            # No improvement or no feature found, break
            break

    # If for some reason best_theta not defined (e.g., no features added)
    if 'best_theta' not in locals():
        # Just train with selected_features (which might be empty)
        X_final, y_final = select_features(train_data, selected_features, target_name)
        X_final = add_intercept(X_final)
        best_theta, _ = logistic_gradient_descent(X_final, y_final, alpha=alpha, num_iterations=num_iterations)

    return selected_features, best_theta, history
