import shutil
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import sys
import sklearn
import xgboost
import joblib
from datetime import datetime
import glob

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# --- Clean old plots ---
for f in glob.glob("*.png"):
    os.remove(f)

# --- Load data ---
data = pd.read_csv("data_tca_clusters_raw.csv")

# --- Feature config ---
categorical_features = ["University", "Program", "Level"]
numeric_features = ["Rent_USD", "Visa_Fee_USD", "Insurance_USD", "Duration_Years"]

target_column = "Total_cost"
X = data[categorical_features + numeric_features]
y_reg = data[target_column]
y_cls = pd.qcut(y_reg, q=3, labels=["Low", "Medium", "High"])

# --- MLflow setup ---
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Education_Cost_Pipeline")

# --- Preprocessing ---
scaling_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features)
])

non_scaling_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
])

# --- Regressors ---
regressors = {
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(objective="reg:squarederror", verbosity=0),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso()
}

# --- Param grids (optional tuning for non-tree models skipped for simplicity) ---
param_grids = {
    "RandomForest": {"regressor__n_estimators": [100], "regressor__max_depth": [10]},
    "GradientBoosting": {"regressor__n_estimators": [100], "regressor__learning_rate": [0.1]},
    "XGBoost": {"regressor__n_estimators": [100], "regressor__max_depth": [5]},
    "LinearRegression": {},
    "Ridge": {"regressor__alpha": [0.1, 1.0]},
    "Lasso": {"regressor__alpha": [0.1, 1.0]}
}

# --- Train/test split ---
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

best_overall_model = None
best_overall_r2 = -np.inf
best_overall_name = ""
predicted_costs = {}
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

performance_log = []

# --- Loop over all models with and without scaling ---
for name, regressor in regressors.items():
    for scaling, preprocessor in [("scaled", scaling_preprocessor), ("unscaled", non_scaling_preprocessor)]:
        run_name = f"{name}_{scaling}_Regressor_{timestamp}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("xgboost_version", xgboost.__version__)

            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", regressor)
            ])

            grid = GridSearchCV(pipe, param_grids[name], cv=5, scoring="r2")
            grid.fit(X_train, y_train_reg)
            best_model = grid.best_estimator_

            mlflow.sklearn.log_model(best_model, "model", input_example=X_test[:1])

            preds = best_model.predict(X_test)
            predicted_costs[run_name] = preds

            mae = mean_absolute_error(y_test_reg, preds)
            mse = mean_squared_error(y_test_reg, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_reg, preds)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            performance_log.append({
                "Model": name,
                "Scaling": scaling,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Best_Params": grid.best_params_
            })

            if r2 > best_overall_r2:
                best_overall_r2 = r2
                best_overall_model = best_model
                best_overall_name = run_name

                plt.figure(figsize=(6, 4))
                plt.scatter(y_test_reg, preds, alpha=0.6)
                plt.xlabel("Actual Total Cost")
                plt.ylabel("Predicted Total Cost")
                plt.title(f"Best Model: {run_name} - Predicted vs Actual")
                z = np.polyfit(y_test_reg, preds, 1)
                p = np.poly1d(z)
                plt.plot(y_test_reg, p(y_test_reg), color="red", linewidth=2, label="Best fit line")
                plt.legend()
                plt.savefig("best_model_plot.png")
                mlflow.log_artifact("best_model_plot.png")
                plt.close()

mlflow.end_run()

joblib.dump(best_overall_model, "model_pipeline.pkl")
print(f"\U0001F4E6 Saved best model ({best_overall_name}) with R² = {best_overall_r2:.3f} to model_pipeline.pkl")

final_preds = predicted_costs[best_overall_name]
pred_tiers = pd.qcut(final_preds, q=3, labels=["Low", "Medium", "High"])

with mlflow.start_run(run_name=f"{best_overall_name}_Classifier_{timestamp}"):
    acc = accuracy_score(y_test_cls, pred_tiers)
    prec = precision_score(y_test_cls, pred_tiers, average="macro")
    rec = recall_score(y_test_cls, pred_tiers, average="macro")
    f1 = f1_score(y_test_cls, pred_tiers, average="macro")
    logloss = log_loss(y_test_cls, pd.get_dummies(pred_tiers))

    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", prec)
    mlflow.log_metric("Recall", rec)
    mlflow.log_metric("F1_score", f1)
    mlflow.log_metric("LogLoss", logloss)

    try:
        auc = roc_auc_score(y_test_cls, pd.get_dummies(pred_tiers), multi_class="ovo", average="macro")
        mlflow.log_metric("AUC_ROC", auc)
    except:
        pass

    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test_cls, pred_tiers, ax=ax)
    plt.title(f"{best_overall_name} - Confusion Matrix")
    file_name = f"confusion_matrix_{best_overall_name}.png"
    plt.savefig(file_name)
    mlflow.log_artifact(file_name)
    plt.close()

result_df = pd.DataFrame({
    "Actual_TCA": y_test_reg.values,
    "Predicted_TCA": final_preds,
    "Actual_Affordability": y_test_cls.values,
    "Predicted_Affordability": pred_tiers
})
result_df.to_csv("model_predictions.csv", index=False)
print("\U0001F4C4 Saved predictions to model_predictions.csv")

performance_df = pd.DataFrame(performance_log)
performance_df.to_csv("model_performance_log.csv", index=False)
print("\U0001F4C4 Saved model performance to model_performance_log.csv")
print("\n\U0001F4CA Classification Report:")
print(classification_report(y_test_cls, pred_tiers))
