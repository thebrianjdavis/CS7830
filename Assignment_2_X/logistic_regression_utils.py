import numpy as np


def sigmoid(z):
    """
    The logistic (sigmoid) function: 1 / (1 + e^{-z}).
    """
    return 1.0 / (1.0 + np.exp(-z))


def logistic_cost_function(X, y, theta):
    """
    Computes the cross-entropy (logistic) cost:
      J(theta) = -1/m * sum[ y*log(h) + (1-y)*log(1-h) ]
    where h = sigmoid(X @ theta).
    """
    m = len(y)
    eps = 1e-10  # prevent log(0)
    h = sigmoid(X.dot(theta))
    cost = - (1.0/m) * np.sum(y * np.log(h + eps) + (1-y) * np.log(1 - h + eps))
    return cost


def logistic_gradient_descent(X, y, alpha=0.01, num_iterations=1000):
    """
    Performs gradient descent for logistic regression (no regularization).
    Returns final theta and cost history for analysis.
    """
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1.0 / m) * X.T.dot(h - y)
        theta -= alpha * gradient

        # Track cost at each iteration
        cost_val = logistic_cost_function(X, y, theta)
        cost_history.append(cost_val)

    return theta, cost_history


def logistic_gradient_descent_regularized(X, y, alpha=0.01, lambda_=1.0, num_iterations=1000):
    """
    L2-regularized gradient descent for logistic regression.
    Regularize all theta_j except intercept (theta[0]).
    """
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1.0 / m) * X.T.dot(h - y)
        # regularization term, skip intercept
        reg = (lambda_ / m) * theta
        reg[0] = 0.0  # no reg on intercept
        gradient += reg

        theta -= alpha * gradient

        cost_val = logistic_cost_function_regularized(X, y, theta, lambda_)
        cost_history.append(cost_val)

    return theta, cost_history


def logistic_cost_function_regularized(X, y, theta, lambda_):
    """
    L2-regularized logistic cost:
      cost = (cross-entropy) + (lambda_/(2m))*sum(theta_j^2 for j>=1).
    """
    m = len(y)
    base_cost = logistic_cost_function(X, y, theta)
    reg_term = (lambda_/(2*m)) * np.sum(theta[1:]**2)
    return base_cost + reg_term


def logistic_cost_function_mse(X, y, theta):
    """
    MSE-based cost for logistic regression (used in Task 4b).
    J(theta) = 1/(2m) * sum( (sigmoid(X@theta) - y)^2 ).
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (1.0/(2*m)) * np.sum((h - y)**2)
    return cost


def logistic_gradient_descent_mse(X, y, alpha=0.01, num_iterations=1000):
    """
    Gradient descent for logistic regression using MSE cost instead
    of cross-entropy.
    """
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        diff = (h - y) * h * (1 - h)  # derivative of MSE wrt theta_j
        gradient = (1.0 / m) * X.T.dot(diff)
        theta -= alpha * gradient

        cost_val = logistic_cost_function_mse(X, y, theta)
        cost_history.append(cost_val)

    return theta, cost_history


def logistic_predict_class(X, theta, threshold=0.5):
    """
    Given X and parameters theta, return binary class predictions (0 or 1)
    by thresholding the sigmoid probabilities at 0.5 by default.
    """
    probs = sigmoid(X.dot(theta))
    return (probs >= threshold).astype(int)


def confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix for binary classification:
      [ [TN, FP],
        [FN, TP] ]
    """
    tn = fp = fn = tp = 0
    for actual, pred in zip(y_true, y_pred):
        if actual == 0 and pred == 0:
            tn += 1
        elif actual == 0 and pred == 1:
            fp += 1
        elif actual == 1 and pred == 0:
            fn += 1
        elif actual == 1 and pred == 1:
            tp += 1
    return np.array([[tn, fp], [fn, tp]])


def accuracy_score(conf_mat):
    """
    Given a 2x2 confusion matrix, returns the accuracy = (TP + TN) / total.
    """
    tn, fp = conf_mat[0]
    fn, tp = conf_mat[1]
    total = tn + fp + fn + tp
    return (tn + tp) / total if total > 0 else 0


def compare_classification_performance(old_metrics, new_metrics, label="Comparison"):
    """
    Compare old vs. new classification performance, printing a short verdict.
    Expects dict with keys: "test_accuracy", etc. (Adapt as needed.)
    """
    old_acc = old_metrics["test_accuracy"]
    new_acc = new_metrics["test_accuracy"]

    print(f"\n=== {label} ===")
    if new_acc > old_acc:
        print("Improved test accuracy.")
    elif new_acc < old_acc:
        print("Worse test accuracy.")
    else:
        print("No change in test accuracy.")
