import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime

def main():
    #  Connect to Hopsworks
    project = hopsworks.login(
        project="CitiBike_Final",     # << your project name
        api_key_value="NoSnqjvqruam2G2e.of4xXCy3fxpjkmgdpJflgRoTRbWkTsXdTM3hlQMGlyU37sXiqgLgGbSyBh57edxq"
    )
    fs = project.get_feature_store()

    #  Load raw dataset (replace this with your real dataset code)
    # For demo I just generate dummy data
    stations = ["8 Ave & W 31 St", "Lafayette St & E 8 St", "Pershing Square North"]
    dates = pd.date_range(end=datetime.today().date(), periods=60)

    data = []
    for station in stations:
        for date in dates:
            trips = np.random.randint(100, 300)
            data.append({
                "start_station_name": station,
                "date": date,
                "trip_count": trips,
                "is_weekend": 1 if date.weekday() >= 5 else 0,
                "day_of_week": date.weekday(),
                "month": date.month
            })

    df = pd.DataFrame(data)
    print(f" Loaded dataset: {df.shape}")

    #  Create (or get) feature group in Hopsworks
    fg = fs.get_or_create_feature_group(
        name="citibike_ts_features",           # << this must match your frontend
        version=1,
        primary_key=["start_station_name", "date"],
        description="Time series features for CitiBike station trips",
        event_time="date"
    )

    #  Insert data into feature store
    fg.insert(df)
    print(f" Data inserted to Hopsworks: {df.shape[0]} rows")

if __name__ == "__main__":
    main()
