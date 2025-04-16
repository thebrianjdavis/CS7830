import numpy as np
from dataloader import select_features, add_intercept
from logistic_regression_utils import (
    logistic_gradient_descent,
    logistic_predict,
    compute_metrics,
    logistic_cost_function,
    # etc.
)


def train_and_evaluate_accuracy(train_data, feature_list, target_name,
                                alpha=0.01, num_iterations=1000, threshold=0.3):
    """
    Trains a logistic model on the chosen features and returns training accuracy.
    (Could also return cost, BIC, or any metric you prefer.)
    """
    X_train, y_train = select_features(train_data, feature_list, target_name)
    X_train = add_intercept(X_train)
    theta, _ = logistic_gradient_descent(X_train, y_train, alpha=alpha, num_iterations=num_iterations)

    preds_train = logistic_predict(X_train, theta, threshold=threshold)
    metrics_train = compute_metrics(y_train, preds_train)

    # returning both accuracy and the final theta
    return metrics_train["accuracy"], theta


def forward_stepwise_selection(train_data, feature_candidates, target_name,
                               max_features=5, alpha=0.01, num_iterations=1000):
    """
    A simple forward selection that picks features one-by-one to maximize training accuracy.
    """
    selected = []
    remaining = feature_candidates[:]
    best_accuracy_so_far = 0.0
    history = []
    best_theta = None

    while len(selected) < max_features and len(remaining) > 0:
        feature_to_add = None
        best_acc_this_round = best_accuracy_so_far

        for f in remaining:
            trial_feats = selected + [f]
            acc, _ = train_and_evaluate_accuracy(
                train_data,
                trial_feats,
                target_name,
                alpha=alpha,
                num_iterations=num_iterations
            )
            if acc > best_acc_this_round:
                best_acc_this_round = acc
                feature_to_add = f

        if feature_to_add is not None:
            selected.append(feature_to_add)
            remaining.remove(feature_to_add)
            best_accuracy_so_far = best_acc_this_round
            # retrain with updated list
            _, local_theta = train_and_evaluate_accuracy(
                train_data, selected, target_name,
                alpha=alpha, num_iterations=num_iterations
            )
            best_theta = local_theta
            history.append({
                "added": feature_to_add,
                "current_accuracy": best_accuracy_so_far,
                "features_set": selected[:]
            })
        else:
            # no improvement
            break

    return selected, best_theta, history


def evaluate_model_on_train_test(train_data, test_data, feature_list, target_name, theta, threshold):
    """
    Evaluate a final logistic model on train and test sets, returning a dict of metrics.
    """
    X_train, y_train = select_features(train_data, feature_list, target_name)
    X_train = add_intercept(X_train)
    train_preds = logistic_predict(X_train, theta, threshold=threshold)
    train_metrics = compute_metrics(y_train, train_preds)

    X_test, y_test = select_features(test_data, feature_list, target_name)
    X_test = add_intercept(X_test)
    test_preds = logistic_predict(X_test, theta, threshold=threshold)
    test_metrics = compute_metrics(y_test, test_preds)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
