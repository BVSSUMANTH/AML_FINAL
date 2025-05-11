#!/usr/bin/env python
"""
Citibike Inference Pipeline
This script fetches the latest data, loads the best model, makes predictions, and saves them.
"""

import pandas as pd
import numpy as np
import hopsworks
import joblib
import os
import logging
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run inference on CitiBike data')
    parser.add_argument(
        '--forecast-days',
        type=int,
        default=7,
        help='Number of days to forecast (default: 7)'
    )
    return parser.parse_args()

def connect_to_hopsworks():
    """
    Connect to Hopsworks and get project.
    """
    logger.info("Connecting to Hopsworks...")
    
    try:
        # Connect to Hopsworks with your credentials
        project = hopsworks.login(
            project="CitiBike_Final",
            api_key_value="NoSnqjvqruam2G2e.of4xXCy3fxpjkmgdpJflgRoTRbWkTsXdTM3hlQMGlyU37sXiqgLgGbSyBh57edxq"
        )
        
        logger.info(f"Successfully connected to Hopsworks project: {project.name}")
        return project
    
    except Exception as e:
        logger.exception(f"Error connecting to Hopsworks: {e}")
        return None

def load_latest_data(project):
    """
    Load the latest data from the feature store.
    """
    logger.info("Loading latest data from feature store...")
    
    try:
        # Get feature store
        fs = project.get_feature_store()
        
        # Get feature group
        fg = fs.get_feature_group(name="citibike_ts_features", version=1)
        
        # Get latest data
        query = fg.select_all()
        df = query.read()
        
        logger.info(f"Successfully loaded latest data: {df.shape}")
        return df
    
    except Exception as e:
        logger.exception(f"Error loading data from feature store: {e}")
        return None

def load_model_from_registry(project):
    """
    Load the best model from the model registry.
    """
    logger.info("Loading model from registry...")
    
    try:
        # Get model registry
        mr = project.get_model_registry()
        
        # Get the model
        model = mr.get_model(name="citibike_trip_predictor", version=1)
        
        # Download the model
        model_dir = model.download()
        logger.info(f"Model downloaded to: {model_dir}")
        
        # Load the model
        model_path = os.path.join(model_dir, "model.pkl")
        predictor = joblib.load(model_path)
        
        # Load feature info
        feature_info_path = os.path.join(model_dir, "feature_info.joblib")
        if os.path.exists(feature_info_path):
            feature_info = joblib.load(feature_info_path)
            logger.info(f"Loaded feature info: {feature_info}")
            return predictor, feature_info
        else:
            logger.warning("Feature info not found, using all features")
            return predictor, None
    
    except Exception as e:
        logger.exception(f"Error loading model from registry: {e}")
        
        # Try loading local model as fallback
        try:
            logger.info("Trying to load local model as fallback...")
            predictor = joblib.load("models/optimized_lgbm_model.joblib")
            feature_info = joblib.load("models/feature_info.joblib")
            logger.info("Loaded local model successfully")
            return predictor, feature_info
        except Exception as e2:
            logger.exception(f"Error loading local model: {e2}")
            return None, None

def prepare_latest_features(df):
    """
    Prepare the latest features for each station.
    """
    logger.info("Preparing latest features for prediction...")
    
    # Get unique stations
    stations = df['start_station_name'].unique()
    logger.info(f"Found {len(stations)} unique stations")
    
    # Get the latest data for each station
    latest_features = {}
    
    for station in stations:
        # Filter data for this station
        station_data = df[df['start_station_name'] == station].sort_values('date')
        
        # Get the latest row
        latest_row = station_data.iloc[-1].to_dict()
        latest_features[station] = latest_row
        
    logger.info(f"Prepared latest features for {len(latest_features)} stations")
    return latest_features

def generate_forecast_dates(days=7):
    """
    Generate dates for the forecast period.
    """
    today = datetime.now().date()
    forecast_dates = [today + timedelta(days=i) for i in range(1, days + 1)]
    return forecast_dates

def make_predictions(predictor, latest_features, forecast_dates, feature_info=None):
    """
    Make predictions for each station for the forecast period.
    """
    logger.info(f"Making predictions for {len(forecast_dates)} days...")
    
    # Dictionary to store predictions
    predictions = {}
    
    for station, latest in latest_features.items():
        station_predictions = []
        
        # Current feature values
        current_features = latest.copy()
        
        # Make predictions for each forecast date
        for i, date in enumerate(forecast_dates):
            # Create feature vector for this prediction
            features = {}
            
            # For the first day, use the latest actual values
            if i == 0:
                for j in range(1, 8):  # Lag 1 to 7
                    features[f'trip_count_lag_{j}'] = current_features.get(f'trip_count_lag_{j}', 0)
                    features[f'avg_duration_lag_{j}'] = current_features.get(f'avg_duration_lag_{j}', 0)
                
                # Rolling features
                for window in [3, 7, 14]:
                    features[f'trip_count_rolling_{window}'] = current_features.get(f'trip_count_rolling_{window}', 0)
                    features[f'avg_duration_rolling_{window}'] = current_features.get(f'avg_duration_rolling_{window}', 0)
            else:
                # Shift lag features using previous predictions
                for j in range(1, 8):  # Lag 1 to 7
                    if j <= i:
                        # Use predicted value
                        lag_idx = i - j
                        features[f'trip_count_lag_{j}'] = station_predictions[lag_idx]['prediction']
                        features[f'avg_duration_lag_{j}'] = current_features.get('avg_duration', 0)  # Use last known avg_duration
                    else:
                        # Use actual value
                        features[f'trip_count_lag_{j}'] = current_features.get(f'trip_count_lag_{j-(i)}', 0)
                        features[f'avg_duration_lag_{j}'] = current_features.get(f'avg_duration_lag_{j-(i)}', 0)
                
                # Simple rolling average calculation (can be improved)
                # Using the predictions we've made so far
                if i >= 3:
                    features['trip_count_rolling_3'] = sum([p['prediction'] for p in station_predictions[i-3:i]]) / 3
                else:
                    features['trip_count_rolling_3'] = current_features.get('trip_count_rolling_3', 0)
                
                if i >= 7:
                    features['trip_count_rolling_7'] = sum([p['prediction'] for p in station_predictions[i-7:i]]) / 7
                else:
                    features['trip_count_rolling_7'] = current_features.get('trip_count_rolling_7', 0)
                
                # For avg_duration rolling, use the last known value
                features['avg_duration_rolling_3'] = current_features.get('avg_duration_rolling_3', 0)
                features['avg_duration_rolling_7'] = current_features.get('avg_duration_rolling_7', 0)
                
                # For 14-day rolling window
                if i >= 14:
                    features['trip_count_rolling_14'] = sum([p['prediction'] for p in station_predictions[i-14:i]]) / 14
                else:
                    features['trip_count_rolling_14'] = current_features.get('trip_count_rolling_14', 0)
                
                features['avg_duration_rolling_14'] = current_features.get('avg_duration_rolling_14', 0)
            
            # Add calendar features
            features['day_of_week'] = date.weekday()
            features['is_weekend'] = 1 if date.weekday() >= 5 else 0
            features['month'] = date.month
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # Select required features if feature_info is provided
            if feature_info and 'features' in feature_info:
                # Make sure all required features are in the DataFrame
                for feature in feature_info['features']:
                    if feature not in features_df.columns:
                        features_df[feature] = 0  # Default value for missing features
                
                # Select only the required features
                features_df = features_df[feature_info['features']]
            
            # Make prediction
            prediction = predictor.predict(features_df)[0]
            
            # Round to nearest integer (can't have fractional trips)
            prediction = max(0, round(prediction))  # Ensure prediction is not negative
            
            # Store prediction
            station_predictions.append({
                'date': date,
                'prediction': prediction,
                'features': features
            })
        
        # Store predictions for this station
        predictions[station] = station_predictions
    
    logger.info(f"Made predictions for {len(predictions)} stations")
    return predictions

def save_predictions(predictions, forecast_days):
    """
    Save predictions to CSV and create visualizations.
    """
    logger.info("Saving predictions...")
    
    # Create directory for predictions
    os.makedirs('predictions', exist_ok=True)
    
    # Create a DataFrame with all predictions
    all_predictions = []
    
    for station, station_predictions in predictions.items():
        for pred in station_predictions:
            all_predictions.append({
                'station': station,
                'date': pred['date'],
                'predicted_trips': pred['prediction']
            })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Save to CSV
    date_str = datetime.now().strftime('%Y%m%d')
    csv_path = f"predictions/citibike_forecast_{date_str}_{forecast_days}days.csv"
    predictions_df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved predictions to: {csv_path}")
    
    # Create visualization
    create_forecast_visualization(predictions_df, forecast_days)
    
    return predictions_df, csv_path

def create_forecast_visualization(predictions_df, forecast_days):
    """
    Create visualization of the forecast.
    """
    logger.info("Creating forecast visualization...")
    
    try:
        # Pivot the data for plotting
        pivot_df = predictions_df.pivot(index='date', columns='station', values='predicted_trips')
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Plot each station
        for station in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[station], marker='o', linewidth=2, label=station)
        
        plt.title(f'CitiBike Trip Forecast ({forecast_days} days)')
        plt.xlabel('Date')
        plt.ylabel('Predicted Number of Trips')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plot_path = f"predictions/citibike_forecast_{datetime.now().strftime('%Y%m%d')}_plot.png"
        plt.savefig(plot_path)
        
        logger.info(f"Saved forecast visualization to: {plot_path}")
        
    except Exception as e:
        logger.exception(f"Error creating visualization: {e}")

def save_to_feature_store(project, predictions_df):
    """
    Save predictions to the feature store.
    """
    logger.info("Saving predictions to feature store...")
    
    try:
        # Get feature store
        fs = project.get_feature_store()
        
        # Define feature group for predictions
        fg = fs.get_or_create_feature_group(
            name="citibike_predictions",
            version=1,
            description="Citibike trip predictions",
            primary_key=["station", "date"],
            event_time="date",
            online_enabled=True
        )
        
        # Ensure date is in datetime format
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        
        # Insert predictions
        fg.insert(predictions_df)
        
        logger.info("Successfully saved predictions to feature store")
        return True
    
    except Exception as e:
        logger.exception(f"Error saving predictions to feature store: {e}")
        return False

def main():
    """
    Main function to execute the inference pipeline.
    """
    logger.info("Starting CitiBike inference pipeline")
    
    try:
        # Parse arguments
        args = parse_arguments()
        forecast_days = args.forecast_days
        logger.info(f"Forecast days: {forecast_days}")
        
        # Connect to Hopsworks
        project = connect_to_hopsworks()
        
        if project is None:
            logger.error("Failed to connect to Hopsworks. Exiting.")
            return
        
        # Load latest data
        df = load_latest_data(project)
        
        if df is None:
            logger.error("Failed to load data. Exiting.")
            return
        
        # Load model
        predictor, feature_info = load_model_from_registry(project)
        
        if predictor is None:
            logger.error("Failed to load model. Exiting.")
            return
        
        # Prepare latest features
        latest_features = prepare_latest_features(df)
        
        # Generate forecast dates
        forecast_dates = generate_forecast_dates(forecast_days)
        
        # Make predictions
        predictions = make_predictions(predictor, latest_features, forecast_dates, feature_info)
        
        # Save predictions
        predictions_df, csv_path = save_predictions(predictions, forecast_days)
        
        # Save to feature store
        save_to_feature_store(project, predictions_df)
        
        logger.info("Inference pipeline completed successfully")
    
    except Exception as e:
        logger.exception(f"Error in inference pipeline: {e}")

if __name__ == "__main__":
    main()