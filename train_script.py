import argparse
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions, squared=False)
    print(f"RMSE: {rmse}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
