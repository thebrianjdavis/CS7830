import numpy as np
from tee import Tee
from dataloader import (
    load_data,
    train_test_split,
    select_features,
    add_intercept,
    remove_rows_with_none,
    feature_scaling
)
from logistic_regression_utils import (
    logistic_gradient_descent,
    logistic_cost_function,
    logistic_predict_class,
    logistic_gradient_descent_regularized,
    logistic_cost_function_mse,
    logistic_gradient_descent_mse,
    confusion_matrix,
    accuracy_score,
    compare_classification_performance
)
from stepwise_regression_logistic import (
    forward_selection_logistic
)

print_cost_debugging = False


def encode_outcome(data_dict, outcome_col="outcome"):
    """
    Encodes the outcome column from 'N'/'Y' to 0/1.
    You can adapt to your dataset's actual categories.
    """
    encoded_vals = []
    for val in data_dict[outcome_col]:
        if val == "N":
            encoded_vals.append(0)
        elif val == "Y":
            encoded_vals.append(1)
        else:
            encoded_vals.append(None)
    data_dict[outcome_col] = encoded_vals
    return data_dict


def run_training():
    # ==========================================================================
    # Task 1: Exploratory Data Analysis
    # ==========================================================================
    print("\n===============================================================")
    print("============ Task 1: Exploratory Data Analysis ===============")
    print("===============================================================")

    # 1. Load data
    data = load_data("datasets/Cancer_dataset1.csv")

    # 1a) Summarize numeric columns
    print("\n1a) Summarizing numeric columns: mean_radius, mean_texture, etc.")
    numeric_cols = [
        "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
        "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points"
    ]
    for col in numeric_cols:
        # We'll just print a quick summary via a small function or your own approach
        col_values = [v for v in data[col] if v is not None]
        if len(col_values) == 0:
            continue
        arr = np.array(col_values)
        print(f"\nSummary for {col}:")
        print(f"  Count: {len(arr)}")
        print(f"  Mean: {np.mean(arr):.3f}")
        print(f"  Std: {np.std(arr, ddof=1):.3f}")
        print(f"  Min: {np.min(arr):.3f}")
        print(f"  25%: {np.percentile(arr, 25):.3f}")
        print(f"  50%: {np.percentile(arr, 50):.3f}")
        print(f"  75%: {np.percentile(arr, 75):.3f}")
        print(f"  Max: {np.max(arr):.3f}")

    # 1b) Summarize categorical variable "outcome"
    print("\n1b) Summarizing the categorical variable 'outcome'...")
    outcomes = data["outcome"]
    outcome_count = len([o for o in outcomes if o is not None])
    unique_vals = list(set([o for o in outcomes if o is not None]))
    freq_dict = {}
    for v in outcomes:
        if v not in (None, ""):
            freq_dict[v] = freq_dict.get(v, 0) + 1
    top_val = None
    top_count = 0
    for k, c in freq_dict.items():
        if c > top_count:
            top_val = k
            top_count = c
    print("Outcome summary:")
    print(f"  Count: {outcome_count}")
    print(f"  Unique Values: {unique_vals}")
    print(f"  Top Value: {top_val}")
    print(f"  Frequency of Top Value: {top_count}")

    # 1c) Encode outcome from categorical to numeric
    print("\n1c) Encoding outcome from N/Y to 0/1 ...")
    data = encode_outcome(data, outcome_col="outcome")
    print("Outcome encoding complete.")

    # 1d) Check for redundant features (example discussion)
    print("\n1d) Checking for redundant features.")
    print("No actual removal done, but you'd remove them if you find e.g. 'id' or 100%-correlated columns.")

    # 1e) Correlation between mean_perimeter and se_perimeter
    print("\n1e) Calculating correlation between mean_perimeter and se_perimeter (if present)...")
    if "se_perimeter" in data:
        # Quick correlation
        mp_vals = []
        sp_vals = []
        for mp, sp in zip(data["mean_perimeter"], data["se_perimeter"]):
            if mp is not None and sp is not None:
                mp_vals.append(mp)
                sp_vals.append(sp)
        if len(mp_vals) > 1:
            corr = np.corrcoef(mp_vals, sp_vals)[0, 1]
            print(f"Correlation(mean_perimeter, se_perimeter) = {corr:.3f}")
        else:
            print("Not enough valid data to compute correlation.")
    else:
        print("se_perimeter column not found.")

    # Remove rows with None in outcome (or other needed columns)
    data = remove_rows_with_none(data, ["outcome"])

    # Split train/test
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)


    # ==========================================================================
    # Task 2: Logistic Regression with One Variable
    # ==========================================================================
    print("\n===============================================================")
    print("====== Task 2: Logistic Regression with One Variable ==========")
    print("===============================================================")

    # 2a) Single variable: mean_area
    print("\n2a) Training logistic regression on 'mean_area' ...")
    X_train_1, y_train_1 = select_features(train_data, ["mean_area"], "outcome")
    X_test_1, y_test_1 = select_features(test_data, ["mean_area"], "outcome")

    # Add intercept
    X_train_1 = add_intercept(X_train_1)
    X_test_1 = add_intercept(X_test_1)

    # Perform gradient descent for logistic cost
    theta_1, cost_history_1 = logistic_gradient_descent(
        X_train_1, y_train_1, alpha=0.001, num_iterations=2000
    )

    if print_cost_debugging:
        print("First 10 cost values (one-feature logistic):", cost_history_1[:10])

    # 2b) Evaluate performance with confusion matrix
    y_pred_train_1 = logistic_predict_class(X_train_1, theta_1)
    y_pred_test_1 = logistic_predict_class(X_test_1, theta_1)

    conf_mat_train_1 = confusion_matrix(y_train_1, y_pred_train_1)
    conf_mat_test_1 = confusion_matrix(y_test_1, y_pred_test_1)

    train_acc_1 = accuracy_score(conf_mat_train_1)
    test_acc_1 = accuracy_score(conf_mat_test_1)

    print("\nConfusion Matrix (Train):", conf_mat_train_1)
    print("Confusion Matrix (Test):", conf_mat_test_1)
    print(f"Train Accuracy: {train_acc_1:.3f}, Test Accuracy: {test_acc_1:.3f}")


    # ==========================================================================
    # Task 3: Logistic Regression with Multiple Variables
    # ==========================================================================
    print("\n===============================================================")
    print("===== Task 3: Logistic Regression with Multiple Variables =====")
    print("===============================================================")

    # 3a) Build logistic regression model using 12 variables
    print("\n3a) Using these 12 features to predict outcome ...")
    features_12 = [
        "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
        "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points",
        "mean_fractal_dimension", "se_perimeter", "se_texture", "se_area"
    ]
    print(features_12)

    X_train_12, y_train_12 = select_features(train_data, features_12, "outcome")
    X_test_12, y_test_12 = select_features(test_data, features_12, "outcome")

    X_train_12 = add_intercept(X_train_12)
    X_test_12 = add_intercept(X_test_12)

    # Perform gradient descent
    theta_12, cost_history_12 = logistic_gradient_descent(
        X_train_12, y_train_12, alpha=0.0001, num_iterations=3000
    )

    y_pred_train_12 = logistic_predict_class(X_train_12, theta_12)
    y_pred_test_12 = logistic_predict_class(X_test_12, theta_12)

    conf_mat_train_12 = confusion_matrix(y_train_12, y_pred_train_12)
    conf_mat_test_12 = confusion_matrix(y_test_12, y_pred_test_12)

    train_acc_12 = accuracy_score(conf_mat_train_12)
    test_acc_12 = accuracy_score(conf_mat_test_12)

    print("\nConfusion Matrix (Train):", conf_mat_train_12)
    print("Confusion Matrix (Test):", conf_mat_test_12)
    print(f"Train Accuracy: {train_acc_12:.3f}, Test Accuracy: {test_acc_12:.3f}")

    # 3b) Forward selection for best subset
    print("\n3b) Forward selection from the same 12 features ...")
    selected_feats_fwd, best_theta_fwd, history_fwd = forward_selection_logistic(
        train_data, "outcome", features_12, alpha=0.0001, num_iterations=3000, max_features=12
    )

    print("Selected features:", selected_feats_fwd)
    print("History of selection steps:")
    for step_info in history_fwd:
        print(step_info)

    # Evaluate forward-selected model
    X_train_sel, y_train_sel = select_features(train_data, selected_feats_fwd, "outcome")
    X_test_sel, y_test_sel = select_features(test_data, selected_feats_fwd, "outcome")

    X_train_sel = add_intercept(X_train_sel)
    X_test_sel = add_intercept(X_test_sel)

    y_pred_train_sel = logistic_predict_class(X_train_sel, best_theta_fwd)
    y_pred_test_sel = logistic_predict_class(X_test_sel, best_theta_fwd)

    conf_mat_train_sel = confusion_matrix(y_train_sel, y_pred_train_sel)
    conf_mat_test_sel = confusion_matrix(y_test_sel, y_pred_test_sel)

    acc_train_sel = accuracy_score(conf_mat_train_sel)
    acc_test_sel = accuracy_score(conf_mat_test_sel)

    print("\nConfusion Matrix (Train):", conf_mat_train_sel)
    print("Confusion Matrix (Test):", conf_mat_test_sel)
    print(f"Train Accuracy: {acc_train_sel:.3f}, Test Accuracy: {acc_test_sel:.3f}")

    # 3c) Compare performance of full model vs. forward subset
    print("\n3c) Comparing 12-feature model vs. forward-selected subset:")
    print(f"Full model test accuracy: {test_acc_12:.3f}")
    print(f"Forward subset test accuracy: {acc_test_sel:.3f}")

    # ==========================================================================
    # Task 4: Experimenting with Regularization and Cost Function
    # ==========================================================================
    print("\n===============================================================")
    print("=== Task 4: Regularization and Alternative Cost Functions ====")
    print("===============================================================")

    # 4a) Regularization and Feature Scaling
    print("\n4a) Regularization and Feature Scaling on best model from 3c...")

    # 4a-I) Regularization
    print("\n4a-I) L2 Regularization on forward-selected model ...")
    lambda_ = 1.0
    theta_reg, cost_hist_reg = logistic_gradient_descent_regularized(
        X_train_sel, y_train_sel, alpha=0.0001, lambda_=lambda_, num_iterations=3000
    )

    y_pred_train_reg = logistic_predict_class(X_train_sel, theta_reg)
    y_pred_test_reg = logistic_predict_class(X_test_sel, theta_reg)

    conf_mat_test_reg = confusion_matrix(y_test_sel, y_pred_test_reg)
    acc_test_reg = accuracy_score(conf_mat_test_reg)

    print("\nConfusion Matrix (Test, Regularized):", conf_mat_test_reg)
    print(f"Test Accuracy (Reg): {acc_test_reg:.3f} vs. Non-reg: {acc_test_sel:.3f}")

    # 4a-II) Feature Scaling
    print("\n4a-II) Feature Scaling using z-score standardization ...")
    # Rebuild X without intercept so we can scale
    X_train_sel_no_int, _ = select_features(train_data, selected_feats_fwd, "outcome")
    X_test_sel_no_int, _ = select_features(test_data, selected_feats_fwd, "outcome")

    X_train_scaled, scale_params = feature_scaling(X_train_sel_no_int)
    means = scale_params["means"]
    stds = scale_params["stds"]
    stds_no_zero = np.where(stds == 0, 1, stds)
    X_test_scaled = (X_test_sel_no_int - means) / stds_no_zero

    X_train_scaled = add_intercept(X_train_scaled)
    X_test_scaled = add_intercept(X_test_scaled)

    theta_scaled, cost_hist_scaled = logistic_gradient_descent(
        X_train_scaled, y_train_sel, alpha=0.01, num_iterations=3000
    )

    y_pred_test_scaled = logistic_predict_class(X_test_scaled, theta_scaled)
    conf_mat_test_scaled = confusion_matrix(y_test_sel, y_pred_test_scaled)
    acc_test_scaled = accuracy_score(conf_mat_test_scaled)

    print("\nConfusion Matrix (Test, Feature-Scaled):", conf_mat_test_scaled)
    print(f"Test Accuracy (Scaled): {acc_test_scaled:.3f} vs. Baseline: {acc_test_sel:.3f}")

    # 4b) Changing the cost function to MSE
    print("\n4b) Changing cost function from cross-entropy to MSE (logistic setup) ...")

    # 4b-I) MSE cost for best model from 3c
    print("4b-I) Fitting logistic with MSE cost ...")
    theta_mse, cost_hist_mse = logistic_gradient_descent_mse(
        X_train_sel, y_train_sel, alpha=0.0001, num_iterations=3000
    )
    y_pred_test_mse = logistic_predict_class(X_test_sel, theta_mse)
    conf_mat_test_mse = confusion_matrix(y_test_sel, y_pred_test_mse)
    acc_test_mse = accuracy_score(conf_mat_test_mse)

    print("\nConfusion Matrix (Test, MSE-based):", conf_mat_test_mse)
    print(f"Test Accuracy (MSE cost): {acc_test_mse:.3f} vs. Cross-Entropy: {acc_test_sel:.3f}")

    print("\n4b-II) Observations: Did MSE cost yield different solution or accuracy?")


def main():
    # We use the same pattern as your linear regression 'main.py'.
    # Tee is used to log output to both console and file if desired.
    with Tee("logistic_output.txt", mode='w'):
        run_training()


# Run main program
if __name__ == '__main__':
    main()
