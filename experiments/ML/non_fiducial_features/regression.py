

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def print_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"MAE : {mae:.4f}")
    

from matplotlib import pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def plot_regression_results(
    y_true,
    y_pred,
    model_name,
    target_name
):

    # Convert to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred

    # ----------------------------------------
    # 1. TRUE VS PREDICTED
    # ----------------------------------------

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.3
    )

    min_val = min(
        y_true.min(),
        y_pred.min()
    )

    max_val = max(
        y_true.max(),
        y_pred.max()
    )

    # Ideal y = x line
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        label="Ideal (y = x)"
    )

    plt.xlabel(
        f"True {target_name} (mmHg)"
    )

    plt.ylabel(
        f"Predicted {target_name} (mmHg)"
    )

    plt.title(
        f"{model_name} — {target_name}: "
        f"True vs Predicted"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ----------------------------------------
    # 2. RESIDUAL VS PREDICTED
    # ----------------------------------------

    plt.figure(figsize=(7, 5))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.3
    )

    # Zero-error line
    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        f"Predicted {target_name} (mmHg)"
    )

    plt.ylabel(
        "Residual (mmHg)"
    )

    plt.title(
        f"{model_name} — {target_name}: "
        f"Residual Plot"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ----------------------------------------
    # 3. RESIDUAL DISTRIBUTION
    # ----------------------------------------

    plt.figure(figsize=(7, 5))

    plt.hist(
        residuals,
        bins=50,
        alpha=0.7
    )

    # Zero-error line
    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Residual (mmHg)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"{model_name} — {target_name}: "
        f"Residual Distribution"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ----------------------------------------
    # 4. MAE VS TRUE TARGET RANGE
    #    WITH ERROR BARS
    # ----------------------------------------

    if target_name.upper() == "DBP":

        bins = [
            50,
            60,
            70,
            80,
            90,
            100,
            120,
            150
        ]

    else:

        # SBP
        bins = [
            70,
            80,
            100,
            120,
            140,
            160,
            180,
            200
        ]


    mae_values = []
    error_std = []
    sample_counts = []
    range_labels = []


    for i in range(len(bins) - 1):

        lower = bins[i]
        upper = bins[i + 1]

        mask = (
            (y_true >= lower) &
            (y_true < upper)
        )

        count = np.sum(mask)

        # Skip range if there are no samples
        if count == 0:
            continue


        # ----------------------------------------
        # Absolute errors for this BP range
        # ----------------------------------------

        absolute_errors = np.abs(
            y_true[mask] -
            y_pred[mask]
        )


        # Mean Absolute Error
        mae = np.mean(
            absolute_errors
        )


        # Standard deviation of absolute errors
        std = np.std(
            absolute_errors
        )


        mae_values.append(mae)
        error_std.append(std)
        sample_counts.append(count)

        range_labels.append(
            f"{lower}–{upper}"
        )


    # ----------------------------------------
    # Plot MAE + error bars
    # ----------------------------------------

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        range_labels,
        mae_values,
        yerr=error_std,
        capsize=5,
        alpha=0.8
    )


    plt.xlabel(
        f"True {target_name} range (mmHg)"
    )

    plt.ylabel(
        "MAE (mmHg)"
    )

    plt.title(
        f"{model_name} — {target_name}: "
        f"MAE by True Target Range"
    )


    # ----------------------------------------
    # Add MAE ± SD and sample count
    # ----------------------------------------

    for bar, mae, std, count in zip(
        bars,
        mae_values,
        error_std,
        sample_counts
    ):

        plt.text(
            bar.get_x() +
            bar.get_width() / 2,

            bar.get_height()+std+1,

            f"{mae:.2f} ± {std:.2f}\n"
            f"(n={count:,})",

            ha="center",
            va="bottom",
            fontsize=9
        )


    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

################################333
# LINEAR REGRESSION

from sklearn.linear_model import LinearRegression

def simple_regression(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name
):
    # 1. Create model
    linear_model = LinearRegression()

    # 2. Train on development data
    linear_model.fit(X_dev, y_dev)

    # 3. Predict on test data
    y_test_pred = linear_model.predict(X_test)

    # 4. Print results
    print("Simple Linear Regression")
    print("------------------------")
    print(f"Target: {target_name}")
    print_metrics(y_test, y_test_pred)

    # 5. Plot results
    plot_regression_results(
        y_test,
        y_test_pred,
        "Linear Regression",
        target_name=target_name
    )

    return linear_model, y_test_pred

def cross_gen(
    X_test_vital,
    y_test_vital,
    linear_model,
    model_name, 
    TARGET
    
):
    y_test_pred = linear_model.predict(X_test_vital)
    print_metrics(y_test_vital, y_test_pred)
    plot_regression_results(
            y_test_vital,
            y_test_pred,
            model_name,
            TARGET
        )



############################################3
#    INTERACTION LINEAR REGRESSION

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


def interaction_regression(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    degree=2,
    cv=3
):
    # 1. Create interaction-only polynomial model
    interaction_model = Pipeline([
        (
            "poly",
            PolynomialFeatures(
                degree=degree,
                interaction_only=True,
                include_bias=False
            )
        ),
        (
            "linear",
            LinearRegression()
        )
    ])

    # 2. Hyperparameter grid
    param_grid = {
        "linear__fit_intercept": [True, False]
    }

    # 3. Grid search with CV
    grid_interaction = GridSearchCV(
        interaction_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=2
    )

    # 4. Train
    grid_interaction.fit(X_dev, y_dev)

    # 5. Print best parameters
    print("Best parameters:")
    print(grid_interaction.best_params_)

    # 6. Predict on untouched test set
    y_test_pred = grid_interaction.predict(X_test)

    # 7. Metrics
    print(f"\nInteraction Linear Regression — {target_name}")
    print("---------------------------------------------")
    print_metrics(y_test, y_test_pred)

    # 8. Plots
    plot_regression_results(
        y_test,
        y_test_pred,
        "Interaction Linear Regression",
        target_name=target_name
    )

    return grid_interaction, y_test_pred



###############################################333
# ROBUST LINEAR REGRESSION
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV


def huber_regression(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    cv=3
):
    # 1. Create Huber Regressor
    robust_model = HuberRegressor(
        max_iter=1000
    )

    # 2. Hyperparameter grid
    param_grid = {
        "epsilon": [1.1, 1.35, 1.5, 2.0],
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]
    }

    # 3. Grid search with cross-validation
    grid_robust = GridSearchCV(
        robust_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=2
    )

    # 4. Train
    grid_robust.fit(X_dev, y_dev)

    # 5. Best parameters
    print("Best parameters:")
    print(grid_robust.best_params_)

    # 6. Predict on test set
    y_test_pred = grid_robust.predict(X_test)

    # 7. Metrics
    print(f"\nRobust Linear Regression — {target_name}")
    print("----------------------------------------")
    print_metrics(y_test, y_test_pred)

    # 8. Plots
    plot_regression_results(
        y_test,
        y_test_pred,
        "Robust Linear Regression",
        target_name=target_name
    )

    return grid_robust, y_test_pred


#######################################33333
# FINE DECISION TREES
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


def decision_tree(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    cv=3
):
    # 1. Create Decision Tree
    fine_tree = DecisionTreeRegressor(
        random_state=42
    )

    # 2. Hyperparameter grid
    param_grid = {
        "max_depth": [10, 15, 20, 25, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    # 3. Grid search with cross-validation
    grid_fine_tree = GridSearchCV(
        fine_tree,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=2
    )

    # 4. Train
    grid_fine_tree.fit(X_dev, y_dev)

    # 5. Best parameters
    print("Best parameters:")
    print(grid_fine_tree.best_params_)

    # 6. Predict on test set
    y_test_pred = grid_fine_tree.predict(X_test)

    # 7. Metrics
    print(f"\nFine Decision Tree — {target_name}")
    print("--------------------------------")
    print_metrics(y_test, y_test_pred)

    # 8. Plots
    plot_regression_results(
        y_test,
        y_test_pred,
        "Fine Decision Tree",
        target_name=target_name
    )

    return grid_fine_tree, y_test_pred


##################################################
# RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def random_forest(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    cv=3
):
    # 1. Create Random Forest
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=2
    )

    # 2. Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"]
    }

    # 3. Grid search with cross-validation
    grid_rf = GridSearchCV(
        rf,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=2
    )

    # 4. Train
    grid_rf.fit(X_dev, y_dev)

    # 5. Best parameters
    print("Best parameters:")
    print(grid_rf.best_params_)

    # 6. Predict on test set
    y_test_pred = grid_rf.predict(X_test)

    # 7. Metrics
    print(f"\nRandom Forest — {target_name}")
    print("----------------------------")
    print_metrics(y_test, y_test_pred)

    # 8. Plots
    plot_regression_results(
        y_test,
        y_test_pred,
        "Random Forest",
        target_name=target_name
    )

    return grid_rf, y_test_pred


###########################################
# EXTRA TREES
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV


def extra_trees(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    cv=3
):
    # 1. Create Extra Trees model
    extra_trees = ExtraTreesRegressor(
        random_state=42,
        n_jobs=2
    )

    # 2. Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": [1.0, "sqrt"]
    }

    # 3. Grid search with cross-validation
    grid_extra_trees = GridSearchCV(
        estimator=extra_trees,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=2
    )

    # 4. Train
    grid_extra_trees.fit(X_dev, y_dev)

    # 5. Best parameters
    print("Best parameters:")
    print(grid_extra_trees.best_params_)

    # 6. Predict on test set
    y_test_pred = grid_extra_trees.predict(X_test)

    # 7. Metrics
    print(f"\nExtra Trees Regressor — {target_name}")
    print("--------------------------------------")
    print_metrics(y_test, y_test_pred)

    # 8. Plots
    plot_regression_results(
        y_test,
        y_test_pred,
        "Extra Trees Regressor",
        target_name=target_name
    )

    return grid_extra_trees, y_test_pred

#########################################33333
# XGBOOST
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


def xgboost_model(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    cv=3
):
    # 1. Create XGBoost model
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2
    )

    # 2. Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # 3. Grid search with cross-validation
    grid_xgb = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=2
    )

    # 4. Train
    grid_xgb.fit(X_dev, y_dev)

    # 5. Best parameters
    print("Best parameters:")
    print(grid_xgb.best_params_)

    # 6. Predict on test set
    y_test_pred = grid_xgb.predict(X_test)

    # 7. Metrics
    print(f"\nXGBoost — {target_name}")
    print("---------------------")
    print_metrics(y_test, y_test_pred)

    # 8. Plots
    plot_regression_results(
        y_test,
        y_test_pred,
        "XGBoost",
        target_name=target_name
    )

    return grid_xgb, y_test_pred


########################################33333
# 
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV


def lightgbm_model(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    cv=3
):
    # 1. Create LightGBM model
    lgbm = LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=2,
        verbosity=-1
    )

    # 2. Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31],
        "max_depth": [-1, 10]
    }

    # 3. Grid search with cross-validation
    grid_lgbm = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=2,
        verbose=2
    )

    # 4. Train
    grid_lgbm.fit(X_dev, y_dev)

    # 5. Best parameters
    print("Best parameters:")
    print(grid_lgbm.best_params_)

    # 6. Predict on test set
    y_test_pred = grid_lgbm.predict(X_test)

    # 7. Metrics
    print(f"\nLightGBM — {target_name}")
    print("------------------------")
    print_metrics(y_test, y_test_pred)

    # 8. Plots
    plot_regression_results(
        y_test,
        y_test_pred,
        "LightGBM",
        target_name=target_name
    )

    return grid_lgbm, y_test_pred



###################################3333333
# MATERN 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    ConstantKernel,
    WhiteKernel
)
from sklearn.model_selection import GridSearchCV
import numpy as np


def gpr_matern(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    n_gpr=5000,
    cv=3
):
    # --------------------------------------------------
    # 1. Take a manageable subset for GPR
    # --------------------------------------------------
    rng = np.random.RandomState(42)

    n_samples = min(n_gpr, len(X_dev))

    indices = rng.choice(
        len(X_dev),
        size=n_samples,
        replace=False
    )

    # Works for both pandas DataFrame and NumPy arrays
    if hasattr(X_dev, "iloc"):
        X_gpr = X_dev.iloc[indices]
    else:
        X_gpr = X_dev[indices]

    if hasattr(y_dev, "iloc"):
        y_gpr = y_dev.iloc[indices]
    else:
        y_gpr = y_dev[indices]

    # --------------------------------------------------
    # 2. Matern 5/2 GPR
    # --------------------------------------------------
    gpr_matern = GaussianProcessRegressor(
        normalize_y=True,
        random_state=42,
        n_restarts_optimizer=1
    )

    # --------------------------------------------------
    # 3. Matern 5/2 kernel grid
    # --------------------------------------------------
    param_grid = {
        "kernel": [
            ConstantKernel(1.0)
            * Matern(
                length_scale=0.5,
                nu=2.5
            )
            + WhiteKernel(
                noise_level=1.0
            ),

            ConstantKernel(1.0)
            * Matern(
                length_scale=1.0,
                nu=2.5
            )
            + WhiteKernel(
                noise_level=1.0
            ),

            ConstantKernel(1.0)
            * Matern(
                length_scale=2.0,
                nu=2.5
            )
            + WhiteKernel(
                noise_level=1.0
            )
        ]
    }

    # --------------------------------------------------
    # 4. GridSearchCV
    # --------------------------------------------------
    grid_gpr_matern = GridSearchCV(
        estimator=gpr_matern,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=1
    )

    # --------------------------------------------------
    # 5. Fit
    # --------------------------------------------------
    grid_gpr_matern.fit(
        X_gpr,
        y_gpr
    )

    # --------------------------------------------------
    # 6. Best parameters
    # --------------------------------------------------
    print("Best parameters:")
    print(grid_gpr_matern.best_params_)

    # --------------------------------------------------
    # 7. Test prediction
    # --------------------------------------------------
    y_test_pred = grid_gpr_matern.predict(X_test)

    # --------------------------------------------------
    # 8. Metrics
    # --------------------------------------------------
    print(f"\nMatern 5/2 GPR — {target_name}")
    print("--------------------------------")

    print_metrics(
        y_test,
        y_test_pred
    )

    # --------------------------------------------------
    # 9. Plots
    # --------------------------------------------------
    plot_regression_results(
        y_test,
        y_test_pred,
        "Matern 5/2 GPR",
        target_name=target_name
    )

    return grid_gpr_matern, y_test_pred


########################################3
# GPR RATIONAL QUADRATIC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RationalQuadratic,
    WhiteKernel
)
from sklearn.model_selection import GridSearchCV


def gpr_rational_quadratic(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    n_gpr=5000,
    cv=3
):
    # --------------------------------------------------
    # 1. Take a manageable subset for GPR
    # --------------------------------------------------
    rng = np.random.RandomState(42)

    n_samples = min(n_gpr, len(X_dev))

    indices = rng.choice(
        len(X_dev),
        size=n_samples,
        replace=False
    )

    # Works for pandas DataFrame and NumPy arrays
    if hasattr(X_dev, "iloc"):
        X_gpr = X_dev.iloc[indices]
    else:
        X_gpr = X_dev[indices]

    if hasattr(y_dev, "iloc"):
        y_gpr = y_dev.iloc[indices]
    else:
        y_gpr = y_dev[indices]

    # --------------------------------------------------
    # 2. Rational Quadratic GPR
    # --------------------------------------------------
    gpr = GaussianProcessRegressor(
        normalize_y=True,
        random_state=42
    )

    # --------------------------------------------------
    # 3. Rational Quadratic kernel grid
    # --------------------------------------------------
    param_grid = {
        "kernel": [
            ConstantKernel(1.0)
            * RationalQuadratic(
                length_scale=1.0,
                alpha=1.0
            )
            + WhiteKernel(
                noise_level=1.0
            ),

            ConstantKernel(1.0)
            * RationalQuadratic(
                length_scale=0.5,
                alpha=1.0
            )
            + WhiteKernel(
                noise_level=1.0
            ),

            ConstantKernel(1.0)
            * RationalQuadratic(
                length_scale=2.0,
                alpha=1.0
            )
            + WhiteKernel(
                noise_level=1.0
            )
        ]
    }

    # --------------------------------------------------
    # 4. GridSearchCV
    # --------------------------------------------------
    grid_gpr_rq = GridSearchCV(
        estimator=gpr,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=1
    )

    # --------------------------------------------------
    # 5. Fit
    # --------------------------------------------------
    grid_gpr_rq.fit(
        X_gpr,
        y_gpr
    )

    # --------------------------------------------------
    # 6. Best parameters
    # --------------------------------------------------
    print("Best parameters:")
    print(grid_gpr_rq.best_params_)

    # --------------------------------------------------
    # 7. Test prediction
    # --------------------------------------------------
    y_test_pred = grid_gpr_rq.predict(X_test)

    # --------------------------------------------------
    # 8. Metrics
    # --------------------------------------------------
    print(f"\nRational Quadratic GPR — {target_name}")
    print("------------------------------------------")

    print_metrics(
        y_test,
        y_test_pred
    )

    # --------------------------------------------------
    # 9. Plots
    # --------------------------------------------------
    plot_regression_results(
        y_test,
        y_test_pred,
        "Rational Quadratic GPR",
        target_name=target_name
    )

    return grid_gpr_rq, y_test_pred



#############################################
## ANN

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np


def ann(
    X_dev,
    y_dev,
    X_test,
    y_test,
    target_name,
    n_ann=5000,
    cv=3
):
    # --------------------------------------------------
    # 1. Take a manageable subset
    # --------------------------------------------------
    rng = np.random.RandomState(42)

    n_samples = min(n_ann, len(X_dev))

    indices = rng.choice(
        len(X_dev),
        size=n_samples,
        replace=False
    )

    # Works for pandas DataFrame and NumPy arrays
    if hasattr(X_dev, "iloc"):
        X_ann = X_dev.iloc[indices]
    else:
        X_ann = X_dev[indices]

    if hasattr(y_dev, "iloc"):
        y_ann = y_dev.iloc[indices]
    else:
        y_ann = y_dev[indices]

    # --------------------------------------------------
    # 2. ANN / MLP
    # --------------------------------------------------
    ann = MLPRegressor(
        random_state=42,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )

    # --------------------------------------------------
    # 3. Hyperparameter grid
    # --------------------------------------------------
    param_grid = {
        "hidden_layer_sizes": [
            (64, 32),
            (128, 64),
            (64, 32, 16)
        ],
        "activation": ["relu"],
        "alpha": [
            0.0001,
            0.001,
            0.01
        ],
        "learning_rate_init": [
            0.001,
            0.01
        ],
        "batch_size": [
            256,
            512
        ]
    }

    # --------------------------------------------------
    # 4. GridSearchCV
    # --------------------------------------------------
    grid_ann = GridSearchCV(
        estimator=ann,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=2,
        verbose=1
    )

    # --------------------------------------------------
    # 5. Fit
    # --------------------------------------------------
    grid_ann.fit(
        X_ann,
        y_ann
    )

    # --------------------------------------------------
    # 6. Best parameters
    # --------------------------------------------------
    print("Best parameters:")
    print(grid_ann.best_params_)

    # --------------------------------------------------
    # 7. Test prediction
    # --------------------------------------------------
    y_test_pred = grid_ann.predict(X_test)

    # --------------------------------------------------
    # 8. Metrics
    # --------------------------------------------------
    print(f"\nANN / MLP Regressor — {target_name}")
    print("------------------------------------")

    print_metrics(
        y_test,
        y_test_pred
    )

    # --------------------------------------------------
    # 9. Plots
    # --------------------------------------------------
    plot_regression_results(
        y_test,
        y_test_pred,
        "ANN / MLP Regressor",
        target_name=target_name
    )

    return grid_ann, y_test_pred