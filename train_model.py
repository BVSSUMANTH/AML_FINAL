! pip install pyarrow

import hopsworks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
import numpy as np
import joblib
import os

def run_training():
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Load features
    fg = fs.get_feature_group(name="citibike_ts_features", version=1)
    df = fg.read()

    X = df[["is_weekend", "day_of_week", "month"]]
    y = df["trip_count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train baseline model
    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f" Model trained. MAE: {mae:.2f} RMSE: {rmse:.2f}")

    # Save model locally
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dummy_model.pkl")
    joblib.dump(model, model_path)

    #  Register to Model Registry
    mr = project.get_model_registry()
    model_registry_obj = mr.python.create_model(
        name="citibike_model",
        metrics={"mae": mae, "rmse": rmse},
        model_dir=model_dir,
        description="Baseline Dummy Model for CitiBike prediction"
    )
    model_registry_obj.save()
    print(" Model saved to Hopsworks Model Registry!")

if __name__ == "__main__":
    run_training()
