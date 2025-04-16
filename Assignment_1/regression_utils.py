import numpy as np


def cost_function(X, y, theta):
    """
    Computes Mean Squared Error (MSE) for predictions given X, y, theta.
    """
    m = len(y)  # num of training examples

    # Compute predictions based on X and theta
    predictions = X.dot(theta)

    # Compute Mean Squared Error, scaled by 1/(2*m)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def gradient_descent(X, y, alpha=0.01, num_iterations=1000):
    """
    Implements Gradient Descent to optimize theta.
    Returns the final theta and history of cost values.
    """
    # Get m (number of samples) and n (number of features)
    m, n = X.shape
    # Initialize parameters to zero
    theta = np.zeros(n)
    # Keep track of cost at each iteration for analysis
    cost_history = []

    for i in range(num_iterations):
        # Compute predictions with current theta
        predictions = X.dot(theta)
        # Calculate errors
        errors = predictions - y
        # Calculate gradient for each theta_j
        gradient = (1 / m) * (X.T.dot(errors))
        # Update parameters by moving opposite the gradient
        theta = theta - alpha * gradient

        # Evaluate cost with the updated theta
        cost_val = cost_function(X, y, theta)
        cost_history.append(cost_val)

        # Print cost to terminal for debugging
        # print(f"Iteration {i+1}/{num_iterations}, Cost: {cost_val}")

    return theta, cost_history


def predict(X, theta):
    """
    Given a feature matrix X and parameters theta, returns predictions.
    """
    # Compute predicted values as dot product of features and parameters
    return X.dot(theta)


def r_squared(y_true, y_pred):
    """
    Computes R^2 statistic.
    """
    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    # Total sum of squares, relative to the mean of y
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # R^2 = 1 - (residual sum of square / total sum of squares)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def adjusted_r_squared(y_true, y_pred, num_features):
    """
    Computes adjusted R^2 statistic.
    """
    # Number of datapointss
    n = len(y_true)
    # Compute base R^2
    r2 = r_squared(y_true, y_pred)
    # Apply the adjustment for the number of features
    # Adjusted R^2 = 1 - ( (1-R^2)*(n-1) / (n - k - 1) )
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - num_features - 1))
    return adj_r2


def bic(y_true, y_pred, num_features):
    """
    Computes the Bayesian Information Criterion (BIC).
    BIC = n * ln(RSS/n) + k * ln(n)
    Where:
    - n is the number of data points
    - RSS is the residual sum of squares
    - k is the number of parameters (features + 1 for intercept)
    """
    # Number of datapoints
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    k = num_features + 1  # +1 for intercept
    # Now use BIC formula
    bic_value = n * np.log(rss / n) + k * np.log(n)
    return bic_value


def gradient_descent_regularized(X, y, alpha=0.01, lambda_=1.0, num_iterations=1000):
    """
    L2-regularized gradient descent
    """
    # Get shape of X, initialize theta as np array of zeroes, initialize list for cost history
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    # Iterate for number of iterations specified
    for i in range(num_iterations):

        # Compute predicted values as dot product of features and parameters
        predictions = X.dot(theta)
        # Calculate errors
        errors = predictions - y

        # For i >= 1 add the regularization term
        # (i=0 is intercept, do not regularize that)
        gradient = (1 / m) * (X.T.dot(errors))

        # Add regularization value for i >= 1
        # Skip index 0 (or do vector approach)
        reg = (lambda_ / m) * theta
        reg[0] = 0.0
        gradient = gradient + reg

        # Recompute theta with step size times gradient
        theta = theta - alpha * gradient

        # Compute regularized cost value and add to list
        cost_val = cost_function_regularized(X, y, theta, lambda_)
        cost_history.append(cost_val)

    return theta, cost_history


def cost_function_regularized(X, y, theta, lambda_):
    """
    Computes the L2-regularized cost function:
    1/(2m) * sum((X*theta - y)^2) + (lambda_/(2m)) * sum(theta_j^2), j>=1
    """
    # Compute the base MSE term from non-regularized cost function
    mse_term = cost_function(X, y, theta)

    m = len(y)  # num training examples

    # Compute regularization term, excluding theta[0] (the intercept)
    reg_term = (lambda_/(2*m)) * np.sum(theta[1:]**2)

    # Final cost is the sum of the MSE term and the L2 regularization penalty
    return mse_term + reg_term


def compare_performance(old_metrics, new_metrics, label="Comparison"):
    """
    Compare old and new Test MSE and Test R^2, print a short verdict.
    old_metrics, new_metrics: dict with keys "test_mse" and "test_r2" at least.
    """
    # Get values from old and new metrics to compare
    old_mse_test = old_metrics['test_mse']
    new_mse_test = new_metrics['test_mse']
    old_r2_test = old_metrics['test_r2']
    new_r2_test = new_metrics['test_r2']

    # Formatted printing logic to assess specified comparison
    print(f"\n=== {label} ===")
    if new_mse_test < old_mse_test and new_r2_test > old_r2_test:
        print("Improved both Test MSE (lower) and Test R^2 (higher).")
    elif new_mse_test < old_mse_test:
        print("Improved Test MSE, but not Test R^2.")
    elif new_r2_test > old_r2_test:
        print("Improved Test R^2, but not Test MSE.")
    else:
        print("No improvement in Test MSE or R^2.")
