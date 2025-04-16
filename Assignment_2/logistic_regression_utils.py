import numpy as np

########################
# HELPER MATH FUNCTIONS
########################


def sigmoid(z):
    """
    Compute the sigmoid function 1 / (1 + exp(-z)).
    Works for scalar or numpy array.
    """
    return 1.0 / (1.0 + np.exp(-z))

########################
# UNREGULARIZED LOGISTIC COST / GRAD
########################


def logistic_cost_function(X, y, theta):
    """
    Binary cross-entropy (logistic) cost:
    J(theta) = -1/m * sum( y*log(h) + (1-y)*log(1-h) )
    where h = sigmoid(X dot theta).
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    # To avoid log(0), clip h
    h = np.clip(h, 1e-12, 1 - 1e-12)
    cost = -(1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    return cost


def logistic_gradient(X, y, theta):
    """
    Gradient of the logistic cost:
    grad_j = (1/m) * sum( (h - y) * x_j )
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = (1.0/m)*X.T.dot(h - y)
    return grad


def logistic_gradient_descent(X, y, alpha=0.01, num_iterations=1000):
    """
    Unregularized gradient descent for logistic regression.
    Returns (theta, cost_history).
    """
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(num_iterations):
        grad = logistic_gradient(X, y, theta)
        theta -= alpha*grad
        cost_val = logistic_cost_function(X, y, theta)
        cost_history.append(cost_val)

        # Optional debug print:
        # if (i+1) % 100 == 0:
        #     print(f"Iteration {i+1}/{num_iterations}, Cost={cost_val:.6f}")
    return theta, cost_history


def logistic_predict(X, theta, threshold=0.3):
    """
    Predicts binary 0 or 1 for each row in X by applying sigmoid
    and thresholding at `threshold`.
    """
    probs = sigmoid(X.dot(theta))
    return (probs >= threshold).astype(int)

########################
# REGULARIZED LOGISTIC COST / GRAD
########################


def logistic_cost_function_reg(X, y, theta, lambda_):
    """
    L2-regularized logistic cost:
    cost = (-1/m)*sum( y*log(h) + (1-y)*log(1-h) ) + (lambda/(2m))*sum(theta[1:]^2)
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    h = np.clip(h, 1e-12, 1 - 1e-12)
    # base logistic cost
    cost_unreg = -(1/m)*np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    # add L2 penalty (not on theta[0])
    reg_term = (lambda_/(2*m))*np.sum(theta[1:]**2)
    return cost_unreg + reg_term


def logistic_gradient_reg(X, y, theta, lambda_):
    """
    Gradient of regularized logistic cost.
    grad_j = (1/m)*sum( (h - y)*x_j ) + (lambda/m)*theta_j  (for j >= 1)
    grad_0 has no regularization.
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = (1.0/m)*X.T.dot(h - y)
    # regularize all but intercept
    reg = (lambda_/m)*theta
    reg[0] = 0.0
    return grad + reg


def logistic_gradient_descent_reg(X, y, alpha=0.01, lambda_=1.0, num_iterations=1000):
    """
    L2-regularized gradient descent for logistic regression.
    """
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    for i in range(num_iterations):
        grad = logistic_gradient_reg(X, y, theta, lambda_)
        theta -= alpha*grad
        cost_val = logistic_cost_function_reg(X, y, theta, lambda_)
        cost_history.append(cost_val)
    return theta, cost_history

########################
# CONFUSION MATRIX & METRICS
########################


def confusion_matrix(y_true, y_pred):
    """
    Returns (TP, FP, FN, TN)
    for binary classification (labels must be 0 or 1).
    """
    tp = fp = fn = tn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def compute_metrics(y_true, y_pred):
    """
    Returns a dictionary with confusion matrix counts plus accuracy.
    """
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

########################
# ALTERNATIVE COST: MSE for logistic
########################


def logistic_cost_function_mse(X, y, theta):
    """
    Mean Squared Error cost for logistic model:
    J_MSE = (1 / 2m) * sum( (h - y)^2 ), where h = sigmoid(X dot theta).
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    mse = (1/(2*m)) * np.sum((h - y)**2)
    return mse


def logistic_gradient_mse(X, y, theta):
    """
    Gradient of the MSE cost w.r.t. theta for logistic regression:
    grad_j = (1/m) * sum( (h - y)*h*(1-h)*x_j )
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    # derivative of (h-y)^2/2 w.r.t. theta = (h-y)*h*(1-h)*x_j
    factor = (h - y)*h*(1-h)  # shape (m,)
    grad = (1.0/m)*X.T.dot(factor)
    return grad


def logistic_gradient_descent_mse(X, y, alpha=0.01, num_iterations=1000):
    """
    Gradient Descent using MSE cost for logistic regression.
    """
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    for i in range(num_iterations):
        grad = logistic_gradient_mse(X, y, theta)
        theta -= alpha*grad
        cost_val = logistic_cost_function_mse(X, y, theta)
        cost_history.append(cost_val)
    return theta, cost_history
