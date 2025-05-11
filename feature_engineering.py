#!/usr/bin/env python
"""
Citibike Feature Engineering Pipeline
This script fetches latest data, processes it, and uploads to Hopsworks Feature Store.
"""

import pandas as pd
import numpy as np
import hopsworks
import requests
import os
import logging
from datetime import datetime, timedelta
import time
import glob

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/feature_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_latest_data():
    """
    Download the latest Citibike trip data.
    Returns a DataFrame or None if download fails.
    """
    # Citibike data URL format (adjust as needed)
    current_date = datetime.now()
    previous_month = current_date.replace(day=1) - timedelta(days=1)
    year_month = previous_month.strftime("%Y%m")
    
    url = f"https://s3.amazonaws.com/tripdata/{year_month}-citibike-tripdata.csv.zip"
    
    logger.info(f"Downloading data from: {url}")
    
    try:
        # Create a temp directory for downloads
        os.makedirs('tmp', exist_ok=True)
        local_path = f"tmp/{year_month}-citibike-tripdata.csv.zip"
        
        # Download the file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Read the CSV
            df = pd.read_csv(local_path)
            logger.info(f"Successfully downloaded data: {df.shape}")
            return df
        else:
            logger.error(f"Failed to download data. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.exception(f"Error downloading data: {e}")
        return None

def preprocess_data(df):
    """
    Clean and preprocess Citi Bike trip data with safe datetime parsing.
    """
    logger.info("Preprocessing data...")
    
    # Convert column names to lowercase and clean
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]
    
    # Handle different column naming conventions
    if 'started_at' in df.columns and 'ended_at' in df.columns:
        df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
        df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
    elif 'starttime' in df.columns and 'stoptime' in df.columns:
        df['started_at'] = pd.to_datetime(df['starttime'], errors='coerce')
        df['ended_at'] = pd.to_datetime(df['stoptime'], errors='coerce')
    
    # Filter out rows with NaN datetime values
    df = df.dropna(subset=['started_at', 'ended_at'])
    
    # Filter out rows with NaN station names
    if 'start_station_name' in df.columns and 'end_station_name' in df.columns:
        df = df.dropna(subset=['start_station_name', 'end_station_name'])
    
    # Calculate trip duration
    df['trip_duration_sec'] = (df['ended_at'] - df['started_at']).dt.total_seconds()
    
    # Filter out very short trips (less than 1 minute)
    df = df[df['trip_duration_sec'] > 60]
    
    # Extract date and time features
    df['date'] = df['started_at'].dt.date
    df['hour'] = df['started_at'].dt.hour
    df['day_of_week'] = df['started_at'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['started_at'].dt.month
    
    logger.info(f"Preprocessing complete. Shape after preprocessing: {df.shape}")
    return df

def get_top_stations(df, n=3):
    """
    Get the top N most popular start stations.
    """
    station_counts = df['start_station_name'].value_counts().head(n)
    logger.info(f"Top {n} Start Stations:\n{station_counts}")
    return station_counts.index.tolist()

def filter_top_stations(df, top_stations):
    """
    Filter dataset to only include trips from top stations.
    """
    filtered_df = df[df['start_station_name'].isin(top_stations)]
    logger.info(f"Filtered data to top stations. New shape: {filtered_df.shape}")
    return filtered_df

def create_time_series_features(df):
    """
    Create time series lag features for each station.
    """
    logger.info("Creating time series features...")
    
    # Group by date and station
    daily_stats = df.groupby(['date', 'start_station_name']).agg(
        trip_count=('trip_duration_sec', 'count'),
        avg_duration=('trip_duration_sec', 'mean')
    ).reset_index()
    
    # Sort by station and date
    daily_stats = daily_stats.sort_values(['start_station_name', 'date'])
    
    # Process each station separately
    stations = daily_stats['start_station_name'].unique()
    all_station_data = []
    
    for station in stations:
        station_data = daily_stats[daily_stats['start_station_name'] == station].copy()
        
        # Create lag features (1 to 28 days)
        for lag in range(1, 29):
            station_data[f'trip_count_lag_{lag}'] = station_data['trip_count'].shift(lag)
            station_data[f'avg_duration_lag_{lag}'] = station_data['avg_duration'].shift(lag)
        
        # Create rolling window features
        for window in [3, 7, 14]:
            station_data[f'trip_count_rolling_{window}'] = station_data['trip_count'].shift(1).rolling(window=window).mean()
            station_data[f'avg_duration_rolling_{window}'] = station_data['avg_duration'].shift(1).rolling(window=window).mean()
        
        all_station_data.append(station_data)
    
    # Combine all station data
    time_series_df = pd.concat(all_station_data, ignore_index=True)
    
    # Drop rows with NaN values (due to lag/rolling features)
    time_series_df = time_series_df.dropna()
    
    logger.info(f"Time series features created. Final shape: {time_series_df.shape}")
    return time_series_df

def upload_to_feature_store(df, feature_group_name):
    """
    Upload the processed data to Hopsworks Feature Store.
    """
    logger.info(f"Connecting to Hopsworks and uploading to feature store: {feature_group_name}")
    
    try:
        # Connect to Hopsworks with your credentials
        project = hopsworks.login(
            project="CitiBike_Final",
            api_key_value="NoSnqjvqruam2G2e.of4xXCy3fxpjkmgdpJflgRoTRbWkTsXdTM3hlQMGlyU37sXiqgLgGbSyBh57edxq"
        )
        
        # Get Feature Store
        fs = project.get_feature_store()
        
        # Convert date column to proper datetime if it's not already
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Define feature group
        feature_group = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=1,
            description="Citibike trip data with time series features",
            primary_key=["start_station_name", "date"],
            event_time="date",
            online_enabled=True
        )
        
        # Upload data to feature store
        feature_group.insert(df)
        
        logger.info(f"Successfully uploaded data to feature store: {feature_group_name}")
        return True
    
    except Exception as e:
        logger.exception(f"Error uploading to Hopsworks: {e}")
        return False

def main():
    """
    Main function to execute the feature engineering pipeline.
    """
    logger.info("Starting CitiBike feature engineering pipeline")
    
    try:
        # Either download new data or use existing data
        new_data = download_latest_data()
        
        if new_data is None:
            # Try to load from local files if download fails
            logger.info("Attempting to load data from local files...")
            csv_files = glob.glob("data/*.csv")
            
            if not csv_files:
                logger.error("No CSV files found and download failed. Exiting.")
                return
            
            logger.info(f"Using local file: {csv_files[0]}")
            new_data = pd.read_csv(csv_files[0])
        
        # Preprocess the data
        processed_data = preprocess_data(new_data)
        
        # Get top stations
        top_stations = get_top_stations(processed_data, n=3)
        
        # Filter to top stations
        filtered_data = filter_top_stations(processed_data, top_stations)
        
        # Create time series features
        time_series_data = create_time_series_features(filtered_data)
        
        # Save processed data locally
        os.makedirs('data', exist_ok=True)
        time_series_data.to_csv(f"data/citibike_ts_features_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        logger.info(f"Saved processed data to local file")
        
        # Upload to feature store
        success = upload_to_feature_store(time_series_data, "citibike_ts_features")
        
        if success:
            logger.info("Feature engineering pipeline completed successfully")
        else:
            logger.error("Feature engineering pipeline completed with errors")
    
    except Exception as e:
        logger.exception(f"Error in feature engineering pipeline: {e}")

if __name__ == "__main__":
    main()