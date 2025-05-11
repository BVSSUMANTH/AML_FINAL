import hopsworks
import pandas as pd
from datetime import datetime, timedelta
import joblib
import numpy as np

def run_inference():
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Load features
    fg = fs.get_feature_group(name="citibike_ts_features", version=1)
    df = fg.read()

    # Predict for latest date
    latest_date = df["date"].max()
    latest_data = df[df["date"] == latest_date]

    X_latest = latest_data[["is_weekend", "day_of_week", "month"]]
    stations = latest_data["start_station_name"].values

    # Load model from Model Registry
    mr = project.get_model_registry()
    model = mr.get_model("citibike_model", version=1)
    model_dir = model.download()
    loaded_model = joblib.load(f"{model_dir}/dummy_model.pkl")

    # Predict
    predictions = loaded_model.predict(X_latest)
    prediction_results = pd.DataFrame({
        "station": stations,
        "date": [datetime.now().date()] * len(stations),
        "predicted_trips": predictions.astype(int)
    })

    print(" Inference done!")
    print(prediction_results)

    #  Save predictions to Hopsworks Feature Store
    predictions_fg = fs.get_or_create_feature_group(
        name="citibike_predictions",
        version=1,
        primary_key=["station", "date"],
        description="Predictions for CitiBike trips"
    )
    predictions_fg.insert(prediction_results)
    print(" Predictions saved to Feature Store!")

if __name__ == "__main__":
    run_inference()
