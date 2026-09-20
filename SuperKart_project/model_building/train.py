
import subprocess
import mlflow
import pandas as pd
import xgboost as xgb
import joblib
import time
from pyngrok import ngrok
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# 1. Kill any process already using port 1558
subprocess.run(["fuser", "-k", "1558/tcp"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-f", "mlflow"], stderr=subprocess.DEVNULL)
time.sleep(1)  # Brief delay to allow port release

# 2. Kill existing ngrok tunnels to avoid duplicate tunnel errors
ngrok.kill()

# 3. Set Ngrok Auth Token
NGROK_AUTH_TOKEN = "3JaaoLCSxHpuLoz0LTvkUetOyfX_2jMkMZCB4kKCAnzgauBaz"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# 4. Start MLflow UI on port 1558
process = subprocess.Popen(["mlflow", "ui", "--port", "1558"])
time.sleep(3)  # Wait for MLflow to finish binding to the port

# 5. Connect ngrok tunnel
public_url = ngrok.connect(1558).public_url
print("MLflow UI is available at:", public_url)

# Set the tracking URL for MLflow
mlflow.set_tracking_uri(public_url)

# Set the name for the experiment
mlflow.set_experiment("SuperKart-Prediction-Experiment")


# NOTE: categorical columns are left as raw strings; they are one-hot-encoded
# inside the pipeline below, so training and serving stay consistent.
df = pd.read_csv("SuperKart_project/data/SuperKart.csv")
df.drop(columns=["Product_Id"], inplace=True)
df.drop(columns=["Store_Id"], inplace=True)


target_col = "Product_Store_Sales_Total"
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = [
    'Product_Weight', 'Product_Allocated_Area', 'Product_MRP'
]
categorical_features = [
    'Product_Sugar_Content', 'Product_Type', 'Store_Establishment_Year',
    'Store_Size', 'Store_Location_City_Type', 'Store_Type'
]


# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define base XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
   'xgbregressor__n_estimators': [50, 75, 100, 125, 150],
    'xgbregressor__max_depth': [2, 3, 4],
    'xgbregressor__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbregressor__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(Xtrain, ytrain)

    # Log nested hyperparameter runs
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predict continuous values directly
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Log regression metrics
    mlflow.log_metrics({
        "train_rmse": mean_squared_error(ytrain, y_pred_train),
        "train_mae": mean_absolute_error(ytrain, y_pred_train),
        "train_r2": r2_score(ytrain, y_pred_train),
        "test_rmse": mean_squared_error(ytest, y_pred_test),
        "test_mae": mean_absolute_error(ytest, y_pred_test),
        "test_r2": r2_score(ytest, y_pred_test)
    })

model_path = "SuperKart_project/deployment/best_SuperKart_model_v1.joblib"
joblib.dump(best_model, model_path)
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved to {model_path}")
