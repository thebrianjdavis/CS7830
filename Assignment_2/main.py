import numpy as np
from tee import Tee
from dataloader import (
    load_data, remove_rows_with_none, stratified_train_test_split,
    select_features, add_intercept, feature_scaling
)
from logistic_regression_utils import (
    logistic_gradient_descent,
    logistic_gradient_descent_reg,
    logistic_gradient_descent_mse,
    logistic_predict,
    compute_metrics,
    logistic_cost_function,
    logistic_cost_function_reg,
    logistic_cost_function_mse
)
from stepwise_regression import (
    forward_stepwise_selection,
    evaluate_model_on_train_test
)

# Global params
ALPHA = 0.00001
THRESHOLD = 0.3
NUM_ITERATIONS = 10000

########################
# EDA Helper Functions
########################


def summarize_numeric_column(column_values):
    cleaned_vals = [v for v in column_values if v is not None]
    arr = np.array(cleaned_vals, dtype=float)
    q25 = np.percentile(arr, 25)
    q50 = np.percentile(arr, 50)
    q75 = np.percentile(arr, 75)
    stats = {
        "count": len(arr),
        "mean": np.mean(arr),
        "std": np.std(arr, ddof=1),
        "min": np.min(arr),
        "25%": q25,
        "50%": q50,
        "75%": q75,
        "max": np.max(arr)
    }
    return stats


def summarize_categorical_column(col_values):
    cleaned_vals = [v for v in col_values if v is not None]
    count = len(cleaned_vals)
    unique_vals = list(set(cleaned_vals))
    freq_map = {}
    for val in cleaned_vals:
        freq_map[val] = freq_map.get(val, 0) + 1
    top_value = None
    top_freq = 0
    for v, f in freq_map.items():
        if f > top_freq:
            top_freq = f
            top_value = v
    summary = {
        "count": count,
        "unique": len(unique_vals),
        "top": top_value,
        "freq": top_freq
    }
    return summary


def pearson_correlation(x_vals, y_vals):
    cleaned = [(x, y) for x,y in zip(x_vals, y_vals) if x is not None and y is not None]
    if not cleaned:
        return 0
    x = np.array([c[0] for c in cleaned], dtype=float)
    y = np.array([c[1] for c in cleaned], dtype=float)
    xbar, ybar = np.mean(x), np.mean(y)
    numerator = np.sum((x - xbar)*(y - ybar))
    denominator = np.sqrt(np.sum((x - xbar)**2)*np.sum((y - ybar)**2))
    if denominator == 0:
        return 0
    return numerator / denominator


def print_class_distribution(data_dict, name="Dataset"):
    """
    Prints how many positives (1) vs. negatives (0)
    and the ratio in the given data dictionary.
    """
    outcomes = data_dict["outcome"]
    total = len(outcomes)
    num_ones = sum(outcomes)  # sum of 1's
    num_zeros = total - num_ones
    ratio_ones = num_ones / total if total > 0 else 0
    ratio_zeros = num_zeros / total if total > 0 else 0

    print(f"{name}:")
    print(f"  Total examples: {total}")
    print(f"  # of positives (R=1): {num_ones} ({ratio_ones:.1%})")
    print(f"  # of negatives (N=0): {num_zeros} ({ratio_zeros:.1%})")
    print()


########################
# MAIN WORKFLOW
########################


def run_logistic_project():
    # 1. Load data
    data = load_data("datasets/Cancer_dataset1.csv")

    # 2. EDA
    #   a) Summaries of numeric variables
    numeric_cols = [
        "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
        "mean_smoothness", "mean_compactness", "mean_concavity",
        "mean_concave_points"
    ]
    print("===== EDA: Numeric Summaries =====")
    for col in numeric_cols:
        stats = summarize_numeric_column(data[col])
        print(f"{col} -> {stats}")
    print()

    #   b) Summaries of the categorical variable "outcome"
    print("===== EDA: Categorical (outcome) =====")
    outcome_summary = summarize_categorical_column(data["outcome"])
    print("outcome ->", outcome_summary)
    print()

    #   c) Encode outcome from N/R to 0/1
    for i, val in enumerate(data["outcome"]):
        if val == "N":
            data["outcome"][i] = 0.0
        else:
            data["outcome"][i] = 1.0

    #   d) (Discussion about redundant features needs to be made here)
    #   e) Correlation
    #      Example correlation: mean_perimeter vs se_perimeter
    if "se_perimeter" in data:
        corr_val = pearson_correlation(data["mean_perimeter"], data["se_perimeter"])
        print(f"Correlation between mean_perimeter and se_perimeter = {corr_val:.4f}")
    print()

    # 3. Clean / Remove rows with None
    columns_of_interest = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area",
                           "mean_smoothness", "mean_compactness", "mean_concavity",
                           "mean_concave_points", "mean_fractal_dimension",
                           "se_perimeter", "se_texture", "se_area", "outcome"]  # trying all
    data = remove_rows_with_none(data, columns_of_interest)

    # 4. Train-Test Split
    train_data, test_data = stratified_train_test_split(
        data, test_size=0.2, random_state=42
    )

    print_class_distribution(train_data, "Train Data")
    print_class_distribution(test_data, "Test Data")

    # ================================
    # 2. LOGISTIC REGRESSION with ONE VARIABLE (mean_area)
    # ================================
    print("===== Logistic Regression with ONE VARIABLE (mean_area) =====")
    X_train_1, y_train_1 = select_features(train_data, ["mean_area"], "outcome")
    X_test_1, y_test_1 = select_features(test_data, ["mean_area"], "outcome")

    X_train_1 = add_intercept(X_train_1)
    X_test_1 = add_intercept(X_test_1)

    theta_1, cost_hist_1 = logistic_gradient_descent(
        X_train_1, y_train_1,
        alpha=ALPHA,
        num_iterations=NUM_ITERATIONS
    )
    print("First 5 cost values:", cost_hist_1[:5])
    print("Last cost value:", cost_hist_1[-1])

    # Evaluate
    train_preds_1 = logistic_predict(X_train_1, theta_1, threshold=THRESHOLD)
    test_preds_1 = logistic_predict(X_test_1, theta_1, threshold=THRESHOLD)

    train_metrics_1 = compute_metrics(y_train_1, train_preds_1)
    test_metrics_1 = compute_metrics(y_test_1, test_preds_1)

    print("Confusion Matrix / Metrics (TRAIN):", train_metrics_1)
    print("Confusion Matrix / Metrics (TEST):", test_metrics_1)
    print()

    # ================================
    # 3. LOGISTIC REGRESSION with MULTIPLE VARIABLES
    # (a) full model with 12 features
    # ================================
    multi_feats = [
        "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
        "mean_smoothness", "mean_compactness", "mean_concavity",
        "mean_concave_points", "mean_fractal_dimension",
        "se_perimeter", "se_texture", "se_area"
    ]

    X_train_multi, y_train_multi = select_features(train_data, multi_feats, "outcome")
    X_test_multi, y_test_multi = select_features(test_data, multi_feats, "outcome")

    X_train_multi = add_intercept(X_train_multi)
    X_test_multi = add_intercept(X_test_multi)

    theta_multi, cost_hist_multi = logistic_gradient_descent(
        X_train_multi, y_train_multi,
        alpha=ALPHA, num_iterations=NUM_ITERATIONS
    )

    print("===== Logistic Regression with MULTIPLE VARIABLES (12) =====")
    print("Final cost:", cost_hist_multi[-1])
    train_preds_multi = logistic_predict(X_train_multi, theta_multi, threshold=0.3)
    test_preds_multi = logistic_predict(X_test_multi, theta_multi, threshold=0.3)

    train_metrics_multi = compute_metrics(y_train_multi, train_preds_multi)
    test_metrics_multi = compute_metrics(y_test_multi, test_preds_multi)

    print("Confusion Matrix / Metrics (TRAIN):", train_metrics_multi)
    print("Confusion Matrix / Metrics (TEST):", test_metrics_multi)
    print()

    # (b) Forward Selection from the same 12 features
    print("===== Forward Selection (accuracy-based) =====")
    selected_feats, best_theta_fs, history_fs = forward_stepwise_selection(
        train_data,
        feature_candidates=multi_feats,
        target_name="outcome",
        max_features=5,
        alpha=ALPHA,
        num_iterations=NUM_ITERATIONS
    )
    print("Selected features (forward):", selected_feats)
    print("History:", history_fs)
    # Evaluate on train & test
    eval_fs = evaluate_model_on_train_test(train_data, test_data, selected_feats, "outcome", best_theta_fs, threshold=THRESHOLD)
    print("Forward Selection - TRAIN:", eval_fs["train_metrics"])
    print("Forward Selection - TEST:", eval_fs["test_metrics"])
    print()

    # (c) Compare full model vs. forward selected
    print("Compare test accuracy or F1 from test_metrics_multi vs. eval_fs['test_metrics'].")
    print()  # <-- TODO

    # ================================
    # 4. REGULARIZATION & FEATURE SCALING
    # ================================
    # (a) For the best model, does regularization help?

    print("===== Regularization (L2) on 12-feature model =====")
    theta_reg, cost_hist_reg = logistic_gradient_descent_reg(
        X_train_multi, y_train_multi,
        alpha=ALPHA, lambda_=1.0, num_iterations=NUM_ITERATIONS
    )
    train_preds_reg = logistic_predict(X_train_multi, theta_reg)
    test_preds_reg = logistic_predict(X_test_multi, theta_reg)

    train_metrics_reg = compute_metrics(y_train_multi, train_preds_reg)
    test_metrics_reg = compute_metrics(y_test_multi, test_preds_reg)

    print("Regularized TRAIN:", train_metrics_reg)
    print("Regularized TEST:", test_metrics_reg)
    print()

    # (a)(ii) Feature Scaling
    print("===== Feature Scaling (12-feature model) =====")
    X_train_noint, _ = select_features(train_data, multi_feats, "outcome")
    X_test_noint, _ = select_features(test_data, multi_feats, "outcome")

    X_train_scaled, scaling_params = feature_scaling(X_train_noint, method="standardize")
    means = scaling_params["means"]
    stds = scaling_params["stds"]
    stds_no_zero = np.where(stds == 0, 1, stds)
    X_test_scaled = (X_test_noint - means) / stds_no_zero

    # add intercept
    X_train_scaled = add_intercept(X_train_scaled)
    X_test_scaled = add_intercept(X_test_scaled)

    theta_scaled, cost_hist_scaled = logistic_gradient_descent(
        X_train_scaled, y_train_multi,
        alpha=ALPHA, num_iterations=NUM_ITERATIONS
    )

    train_preds_scaled = logistic_predict(X_train_scaled, theta_scaled)
    test_preds_scaled = logistic_predict(X_test_scaled, theta_scaled)
    train_metrics_scaled = compute_metrics(y_train_multi, train_preds_scaled)
    test_metrics_scaled = compute_metrics(y_test_multi, test_preds_scaled)

    print("Scaled Model TRAIN:", train_metrics_scaled)
    print("Scaled Model TEST:", test_metrics_scaled)
    print()

    # (b) Changing cost to MSE
    print("===== MSE Cost for Logistic Regression (12 features) =====")
    theta_mse, cost_hist_mse = logistic_gradient_descent_mse(
        X_train_multi, y_train_multi,
        alpha=ALPHA, num_iterations=NUM_ITERATIONS
    )

    train_preds_mse = logistic_predict(X_train_multi, theta_mse)
    test_preds_mse = logistic_predict(X_test_multi, theta_mse)

    train_metrics_mse = compute_metrics(y_train_multi, train_preds_mse)
    test_metrics_mse = compute_metrics(y_test_multi, test_preds_mse)
    print("MSE-based TRAIN:", train_metrics_mse)
    print("MSE-based TEST:", test_metrics_mse)
    print("Compare to cross-entropy-based results to see if there is any difference.")


def main():
    with Tee("logistic_output.txt", mode="w"):
        run_logistic_project()


if __name__ == "__main__":
    main()
