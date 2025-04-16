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
from regression_utils import (
    gradient_descent,
    predict,
    r_squared,
    adjusted_r_squared,
    gradient_descent_regularized,
    compare_performance
)
from stepwise_regression import (
    forward_stepwise,
    backward_stepwise,
    evaluate_model_on_train_test
)

print_cost_debugging = False


def run_training():
    # Preprocessing Load and Clean Data
    data = load_data('datasets/Cancer_dataset.csv')
    columns_of_interest = ['mean_texture', 'tumor_size', 'lymph_node_status']
    data = remove_rows_with_none(data, columns_of_interest)

    # Preprocessing Train-Test Split
    train_data, test_data = train_test_split(
        data, test_size=0.2, shuffle=True, random_state=42
    )

    # ==========================================================================
    # Task 1: Single-Feature Linear Regression
    # ==========================================================================

    # Set features for train and test
    X_train_1, y_train_1 = select_features(train_data, ['mean_texture'], 'tumor_size')
    X_test_1, y_test_1 = select_features(test_data, ['mean_texture'], 'tumor_size')

    # Add intercept
    X_train_1 = add_intercept(X_train_1)
    X_test_1 = add_intercept(X_test_1)

    # Perform gradient descent
    theta_1, cost_hist_1 = gradient_descent(
        X_train_1, y_train_1, alpha=0.001, num_iterations=2000
    )

    # Prints the first 10 cost values for debugging purposes
    if print_cost_debugging:
        print("First 10 cost values (one-feature model):", cost_hist_1[:10])

    # Evaluate (make predictions for task 1)
    pred_train_1 = predict(X_train_1, theta_1)
    pred_test_1 = predict(X_test_1, theta_1)

    # Calculate metrics for task 1
    mse_train_1 = np.mean((pred_train_1 - y_train_1)**2)
    mse_test_1 = np.mean((pred_test_1 - y_test_1)**2)
    r2_train_1 = r_squared(y_train_1, pred_train_1)
    r2_test_1 = r_squared(y_test_1, pred_test_1)
    adj_r2_train_1 = adjusted_r_squared(y_train_1, pred_train_1, num_features=1)
    adj_r2_test_1 = adjusted_r_squared(y_test_1, pred_test_1, num_features=1)

    # Print task 1 metrics for report
    print("\n===============================================================")
    print("============ Task 1: Single Feature (mean_texture) ============")
    print("===============================================================")
    print(f"Train MSE: {mse_train_1:.3f}, Test MSE: {mse_test_1:.3f}")
    print(f"Train R^2: {r2_train_1:.3f}, Test R^2: {r2_test_1:.3f}")
    print(f"Train Adj R^2: {adj_r2_train_1:.3f}, Test Adj R^2: {adj_r2_test_1:.3f}")

    # My notes for task 1
    print("\nTask 1: Single Feature Notes")
    print("There aren't many notes for the first task, given that it is ")
    print("establishing a baseline for the assignment. The negative R^2 ")
    print("and Adj R^2 scores on both the train and test sets suggests ")
    print("that mean_texture by itself has weak linear correlation with ")
    print("tumor_size in this dataset. Thus the single feature isn't a ")
    print("good predictor for this purpose.")

    # Store task 1 metrics for comparison
    model_1_metrics = {
        "train_mse": mse_train_1,
        "test_mse": mse_test_1,
        "train_r2": r2_train_1,
        "test_r2": r2_test_1
    }

    # ==========================================================================
    # Task 2: Two-Feature Linear Regression
    # ==========================================================================

    # Set features for train and test
    X_train_2, y_train_2 = select_features(train_data, ['mean_texture', 'lymph_node_status'], 'tumor_size')
    X_test_2, y_test_2 = select_features(test_data, ['mean_texture', 'lymph_node_status'], 'tumor_size')

    # Add intercept
    X_train_2 = add_intercept(X_train_2)
    X_test_2 = add_intercept(X_test_2)

    # Perform gradient descent
    theta_2, cost_hist_2 = gradient_descent(
        X_train_2, y_train_2, alpha=0.001, num_iterations=2000
    )

    # Prints the first 10 cost values for debugging purposes
    if print_cost_debugging:
        print("First 10 cost values (two-feature model):", cost_hist_2[:10])

    # Evaluate (make predictions for task 2)
    pred_train_2 = predict(X_train_2, theta_2)
    pred_test_2 = predict(X_test_2, theta_2)

    # Calculate metrics for task 2
    mse_train_2 = np.mean((pred_train_2 - y_train_2)**2)
    mse_test_2 = np.mean((pred_test_2 - y_test_2)**2)
    r2_train_2 = r_squared(y_train_2, pred_train_2)
    r2_test_2 = r_squared(y_test_2, pred_test_2)
    adj_r2_train_2 = adjusted_r_squared(y_train_2, pred_train_2, num_features=2)
    adj_r2_test_2 = adjusted_r_squared(y_test_2, pred_test_2, num_features=2)

    # Print task 2 metrics
    print("\n===============================================================")
    print("=== Task 2: Two Features (mean_texture + lymph_node_status) ===")
    print("===============================================================")
    print(f"Train MSE: {mse_train_2:.3f}, Test MSE: {mse_test_2:.3f}")
    print(f"Train R^2: {r2_train_2:.3f}, Test R^2: {r2_test_2:.3f}")
    print(f"Train Adj R^2: {adj_r2_train_2:.3f}, Test Adj R^2: {adj_r2_test_2:.3f}")

    # My notes for task 2
    print("\nTask 2: Two-Feature Notes")
    print("Using two features definitely performed better than one. Lower ")
    print("MSE in both train and test shows that this model is fitting the ")
    print("data better than using the single feature as in task 1. In this ")
    print("task R^2 was positive in the train set, which explains some ")
    print("variance, but R^2 was negative in the test set (so it is under-")
    print("performing on this subset). The negative R^2 on the test set ")
    print("implies that the model is still not strongly predictive.")

    # Store task 2 metrics for comparison
    model_2_metrics = {
        "train_mse": mse_train_2,
        "test_mse": mse_test_2,
        "train_r2": r2_train_2,
        "test_r2": r2_test_2
    }

    # Compare Task 1 single-feature vs. Task 2 two-feature
    print("\nDid task 2's model show improvement on the usage of a single ")
    print("feature?")
    compare_performance(model_1_metrics, model_2_metrics, label="Adding 'lymph_node_status'")

    # ==========================================================================
    # Task 3: Stepwise Regression (Forward and Backward)
    # ==========================================================================

    # 3a. Forward Stepwise with up to 5 features from these 10
    features_for_forward = [
        # 'mean_radius',
        'mean_perimeter',
        # 'mean_area',
        'mean_smoothness',
        # 'mean_symmetry',
        'mean_fractal_dimension',
        # 'worst_radius',
        'worst_area',
        # 'worst_symmetry',
        'lymph_node_status'
    ]

    # Perform forward stepwise regression
    selected_feats_fwd, best_theta_fwd, history_fwd = forward_stepwise(
        train_data=train_data,
        feature_candidates=features_for_forward,
        target_name='tumor_size',
        max_features=5,
        alpha=1e-5,
        num_iterations=10000,
        criterion='bic'
    )

    # Print metrics for task 3a
    print("\n===============================================================")
    print("============ Task 3a: Forward Stepwise Regression =============")
    print("===============================================================")
    print("Selected Features (Forward):", selected_feats_fwd)
    print("History of additions:")

    # Print what features are added in task 3a
    for record in history_fwd:
        print(record)

    # Calculate metrics by evaluating test set after forward stepwise train
    fwd_metrics = evaluate_model_on_train_test(
        train_data, test_data,
        selected_feats_fwd, 'tumor_size',
        best_theta_fwd
    )

    # Print the actual metrics train vs test for 3a
    print(f"Forward Stepwise - Train MSE: {fwd_metrics['train_mse']:.3f}, Test MSE: {fwd_metrics['test_mse']:.3f}")
    print(f"Forward Stepwise - Train R^2: {fwd_metrics['train_r2']:.3f}, Test R^2: {fwd_metrics['test_r2']:.3f}")
    print(f"Forward Stepwise - Train Adj R^2: {fwd_metrics['train_adj_r2']:.3f}, Test Adj R^2: {fwd_metrics['test_adj_r2']:.3f}")

    # 3b. Backward Stepwise (start with all 10, remove up to 5)
    features_for_backward = [
        'mean_radius',
        'mean_perimeter',
        'mean_area',
        'mean_smoothness',
        'mean_symmetry',
        'mean_fractal_dimension',
        'worst_radius',
        'worst_area',
        'worst_symmetry',
        'lymph_node_status'
    ]

    # Perform backward stepwise regression for 3b
    selected_feats_bwd, best_theta_bwd, history_bwd = backward_stepwise(
        train_data=train_data,
        feature_candidates=features_for_backward,
        target_name='tumor_size',
        num_features_to_remove=5,
        alpha=1e-5,
        num_iterations=10000,
        criterion='bic'
    )

    # Print metrics for task 3b
    print("\n===============================================================")
    print("============ Task 3b: Backward Stepwise Regression ============")
    print("===============================================================")
    print("Selected Features (Backward):", selected_feats_bwd)
    print("History of removals:")

    # Print what features are removed in task 3b
    for record in history_bwd:
        print(record)

    # Calculate metrics by evaluating test set after backward stepwise train
    bwd_metrics = evaluate_model_on_train_test(
        train_data, test_data,
        selected_feats_bwd, 'tumor_size',
        best_theta_bwd
    )

    # Print the actual metrics train vs test for 3b
    print(f"Backward Stepwise - Train MSE: {bwd_metrics['train_mse']:.3f}, Test MSE: {bwd_metrics['test_mse']:.3f}")
    print(f"Backward Stepwise - Train R^2: {bwd_metrics['train_r2']:.3f}, Test R^2: {bwd_metrics['test_r2']:.3f}")
    print(f"Backward Stepwise - Train Adj R^2: {bwd_metrics['train_adj_r2']:.3f}, Test Adj R^2: {bwd_metrics['test_adj_r2']:.3f}")

    # 3c: Compare final models from forward vs. backward
    print("\n===============================================================")
    print("======== Task 3c: Compare Forward vs. Backward Models =========")
    print("===============================================================")
    print("\nDid the backward linear regression model perform better than ")
    print("the forward linear regression?")
    compare_performance(fwd_metrics, bwd_metrics, label="Forward vs. Backward Stepwise")

    # 3d: Compare final stepwise model with the two-feature model from Task 2
    print("\n===============================================================")
    print("====== Task 3d: Compare Stepwise with Two-Feature Model =======")
    print("===============================================================")
    print("\nDid task 3's models show improvement on the two-feature linear ")
    print("regression model?")
    compare_performance(model_2_metrics, fwd_metrics, label="Two-Feature vs. Forward Stepwise")
    compare_performance(model_2_metrics, bwd_metrics, label="Two-Feature vs. Backward Stepwise")

    # My notes for task 3
    print("\nTask 3: Stepwise Linear Regression Notes")
    print("The forward stepwise process determined that mean_perimeter and ")
    print("lymph_node_status gave the best improvement in terms of BIC for ")
    print("each iteration. The test MSE showed an improvement over both ")
    print("test MSE scores in task 1 and task 2. The test R^2 was still ")
    print("negative, but less negative than in task 1 or 2's results, so it ")
    print("is improving. So forward stepwise outperforms both earlier, ")
    print("simpler models. The backward stepwise process ended up with all ")
    print("10 of its features still listed, but the MSE and R^2 values are ")
    print("NaN, indicating that the model diverged and was unable to ")
    print("produce a stable result. In the comparison for forward stepwise ")
    print("and backward stepwise, backward cannot show any improvement ")
    print("since it didn't produce a result. Same with backward stepwise ")
    print("and the two-feature model.")

    # ==========================================================================
    # Task 4: Regularization & Feature Scaling
    # ==========================================================================
    print("\n===============================================================")
    print("================= Task 4a: L2 Regularization ==================")
    print("===============================================================")
    # Take the forward stepwise model from task 3d as baseline for comparison
    baseline_metrics_t3 = fwd_metrics

    # Rebuild X, y for the new model
    X_train_best, y_train_best = select_features(train_data, selected_feats_fwd, 'tumor_size')
    X_test_best, y_test_best = select_features(test_data, selected_feats_fwd, 'tumor_size')
    X_train_best = add_intercept(X_train_best)
    X_test_best = add_intercept(X_test_best)

    lambda_ = 1.0

    # Perform regularized gradient descent with unscaled values
    theta_reg, cost_hist_reg = gradient_descent_regularized(
        X_train_best, y_train_best,
        alpha=1e-5,
        lambda_=lambda_,
        num_iterations=5000  # trying lower num iter here
    )

    # Calculate metrics for task 4a
    y_pred_train_reg = predict(X_train_best, theta_reg)
    y_pred_test_reg = predict(X_test_best, theta_reg)
    mse_train_reg = np.mean((y_pred_train_reg - y_train_best)**2)
    mse_test_reg = np.mean((y_pred_test_reg - y_test_best)**2)
    r2_train_reg = r_squared(y_train_best, y_pred_train_reg)
    r2_test_reg = r_squared(y_test_best, y_pred_test_reg)

    # Store task 4a metrics for comparison
    reg_metrics = {
        "train_mse": mse_train_reg,
        "test_mse": mse_test_reg,
        "train_r2": r2_train_reg,
        "test_r2": r2_test_reg
    }

    # Print task 4a metrics
    print(f"Regularized Model (lambda={lambda_}):")
    print(f"Train MSE: {mse_train_reg:.3f}, Test MSE: {mse_test_reg:.3f}")
    print(f"Train R^2: {r2_train_reg:.3f}, Test R^2: {r2_test_reg:.3f}")

    # Compare task 4a to best performing from 3d
    compare_performance(baseline_metrics_t3, reg_metrics, label="L2 Regularization vs. Baseline")

    print("\n===============================================================")
    print("================== Task 4b: Feature Scaling ===================")
    print("===============================================================")
    # Scale the same selected_feats_fwd
    X_train_best_no_int, y_train_best = select_features(train_data, selected_feats_fwd, 'tumor_size')
    X_test_best_no_int, y_test_best = select_features(test_data, selected_feats_fwd, 'tumor_size')

    # Scale using z-score standardize -- scaling_params will store mean and std dev on best training w/o intercept
    X_train_best_scaled, scaling_params = feature_scaling(X_train_best_no_int, method="standardize")

    # Retrieve means and std devs
    means = scaling_params["means"]
    stds = scaling_params["stds"]
    stds_no_zero = np.where(stds == 0, 1, stds)  # std devs exclude zero vals
    X_test_best_scaled = (X_test_best_no_int - means) / stds_no_zero  # calc scaled vals

    # Add intercept after scaling
    X_train_best_scaled = add_intercept(X_train_best_scaled)
    X_test_best_scaled = add_intercept(X_test_best_scaled)

    # Retrain using normal gradient descent on scaled data to allow for comparison of both
    theta_scaled, cost_hist_scaled = gradient_descent(
        X_train_best_scaled, y_train_best,
        alpha=0.01,
        num_iterations=2000
    )

    # Calculate metrics for task 4b
    y_pred_train_scaled = predict(X_train_best_scaled, theta_scaled)
    y_pred_test_scaled = predict(X_test_best_scaled, theta_scaled)
    mse_train_scaled = np.mean((y_pred_train_scaled - y_train_best)**2)
    mse_test_scaled = np.mean((y_pred_test_scaled - y_test_best)**2)
    r2_train_scaled = r_squared(y_train_best, y_pred_train_scaled)
    r2_test_scaled = r_squared(y_test_best, y_pred_test_scaled)

    # Store task 4b metrics for comparison
    scaled_metrics = {
        "train_mse": mse_train_scaled,
        "test_mse": mse_test_scaled,
        "train_r2": r2_train_scaled,
        "test_r2": r2_test_scaled
    }

    # Print task 4b metrics
    print("Scaled Model (no regularization):")
    print(f"Train MSE: {mse_train_scaled:.3f}, Test MSE: {mse_test_scaled:.3f}")
    print(f"Train R^2: {r2_train_scaled:.3f}, Test R^2: {r2_test_scaled:.3f}")

    # Compare metrics for tasks 4a vs 4b
    compare_performance(baseline_metrics_t3, scaled_metrics, label="Feature Scaling vs. Baseline")

    # Not explicitly specified in assignment whether regularization and feature scaling
    # should be assessed separately or together for 4b -- I chose to perform each
    # individually and then do both scaling and regularization together here to see how
    # each affected performance separately as well as together

    # Do "Scaled + Regularized" (alt 4b)
    lambda_scaled = 1.0

    # Perform regularized gradient descent with scaled values
    theta_scaled_reg, cost_hist_scaled_reg = gradient_descent_regularized(
        X_train_best_scaled, y_train_best,
        alpha=0.01,
        lambda_=lambda_scaled,
        num_iterations=2000
    )

    # Calculate metrics after regularized gradient descent on scaled values
    y_pred_train_scaled_reg = predict(X_train_best_scaled, theta_scaled_reg)
    y_pred_test_scaled_reg = predict(X_test_best_scaled, theta_scaled_reg)
    mse_train_scaled_reg = np.mean((y_pred_train_scaled_reg - y_train_best)**2)
    mse_test_scaled_reg = np.mean((y_pred_test_scaled_reg - y_test_best)**2)
    r2_train_scaled_reg = r_squared(y_train_best, y_pred_train_scaled_reg)
    r2_test_scaled_reg = r_squared(y_test_best, y_pred_test_scaled_reg)

    # Store scaled+reg metrics for comparison
    scaled_reg_metrics = {
        "train_mse": mse_train_scaled_reg,
        "test_mse": mse_test_scaled_reg,
        "train_r2": r2_train_scaled_reg,
        "test_r2": r2_test_scaled_reg
    }

    # Print scaled+reg metrics
    print(f"\nScaled + Regularized (lambda={lambda_scaled}):")
    print(f"Train MSE: {mse_train_scaled_reg:.3f}, Test MSE: {mse_test_scaled_reg:.3f}")
    print(f"Train R^2: {r2_train_scaled_reg:.3f}, Test R^2: {r2_test_scaled_reg:.3f}")

    # Compare metrics for tasks 4a+4b vs baseline
    compare_performance(baseline_metrics_t3, scaled_reg_metrics, label="Scaled+Regularized vs. Baseline")
    # Compare combined 4a+4b vs just 4b
    compare_performance(scaled_metrics, scaled_reg_metrics, label="Scaled+Regularized vs. Feature Scaling")

    # My notes for task 4
    print("\nTask 4: Regularization and Featuring Scaling Notes")
    print("As the comparison logs showed, test MSE and R^2 continued to ")
    print("improve relative to the baseline for forward stepwise regression.")
    print("The R^2 was the highest (least negative), so regularization ")
    print("still slightly improved the generalization. Feature scaling ")
    print("performed slightly worse than regularization in terms of both ")
    print("MSE and R^2. Interestingly, the combination of regularization ")
    print("and feature scaling performed worse than just regularization on ")
    print("the forward stepwise regression model. The MSE and R^2 for the ")
    print("combined regularization and feature scaling model was still ")
    print("better than the feature scaling model alone. So overall the best ")
    print("performing ended up being forward stepwise regression with ")
    print("regularization.")


def main():
    # Established a decorator similar to Unix tee to write to file
    with Tee("model_output.txt", mode='w'):
        run_training()


# Run main program
if __name__ == '__main__':
    main()
