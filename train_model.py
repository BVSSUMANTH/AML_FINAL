#!/usr/bin/env python
"""
Simplified Citibike Model Training Pipeline
This script retrieves feature data from Hopsworks, trains a simple model, and registers it.
"""

import pandas as pd
import numpy as np
import hopsworks
import joblib
import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data_from_feature_store():
    """
    Load the latest data from Hopsworks Feature Store.
    """
    logger.info("Loading data from Hopsworks Feature Store...")
    
    try:
        # Connect to Hopsworks with your credentials
        project = hopsworks.login(
            project="CitiBike_Final",
            api_key_value="NoSnqjvqruam2G2e.of4xXCy3fxpjkmgdpJflgRoTRbWkTsXdTM3hlQMGlyU37sXiqgLgGbSyBh57edxq"
        )
        
        # Get Feature Store
        fs = project.get_feature_store()
        
        # Get the feature group
        fg = fs.get_feature_group(name="citibike_ts_features", version=1)
        
        # Get the data
        df = fg.read()
        
        logger.info(f"Successfully loaded data: {df.shape}")
        return df, project
    
    except Exception as e:
        logger.exception(f"Error loading data from Hopsworks: {e}")
        return None, None

def time_series_train_test_split(df, test_size=0.2):
    """
    Perform time-based train/test split for time series data.
    """
    logger.info("Performing time-based train/test split...")
    
    # Sort by date
    df = df.sort_values("date")
    
    # Determine cutoff point
    cutoff_idx = int(len(df) * (1 - test_size))
    cutoff_date = df.iloc[cutoff_idx]["date"]
    
    # Split data
    train_df = df[df["date"] < cutoff_date]
    test_df = df[df["date"] >= cutoff_date]
    
    logger.info(f"Train set: {train_df.shape}, Test set: {test_df.shape}")
    
    return train_df, test_df

def train_model(X_train, y_train, X_test, y_test):
    """
    Train a simple RandomForest model.
    """
    logger.info("Training RandomForest model...")
    
    # Create directory for models
    os.makedirs('models', exist_ok=True)
    
    # Train model with limited number of trees for speed
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info(f"Model MAE: {mae:.2f}")
    
    # Save model locally
    model_path = "models/rf_model.joblib"
    joblib.dump(model, model_path)
    
    return model, mae, model_path

def register_model(project, model_path, model_name, mae):
    """
    Register the model in Hopsworks Model Registry.
    """
    logger.info(f"Registering model to Hopsworks Model Registry...")
    
    try:
        # Get model registry
        mr = project.get_model_registry()
        
        # Create model in registry
        model = mr.python.create_model(
            name=model_name,
            metrics={"mae": mae},
            description=f"CitiBike trip prediction model (MAE: {mae:.2f})"
        )
        
        # Upload the model file
        model.save(model_path)
        
        logger.info(f"Successfully registered model: {model_name}")
        return model
    
    except Exception as e:
        logger.exception(f"Error registering model: {e}")
        return None

def deploy_model(project, model_name):
    """
    Deploy the model to Hopsworks serving.
    """
    logger.info(f"Deploying model {model_name} to Hopsworks serving...")
    
    try:
        # Get model registry
        mr = project.get_model_registry()
        
        # Get the model
        model = mr.get_model(name=model_name, version=1)
        
        # Get or create serving instance
        serving = project.get_model_serving()
        
        # Deploy the model
        deployment = serving.create_deployment(
            name=model_name,
            model=model
        )
        
        logger.info(f"Model deployed successfully: {model_name}")
        return True
    
    except Exception as e:
        logger.exception(f"Error deploying model: {e}")
        return False

def main():
    """
    Main function to execute the model training pipeline.
    """
    logger.info("Starting simplified CitiBike model training pipeline")
    
    try:
        # Load data from Hopsworks
        df, project = load_data_from_feature_store()
        
        if df is None or project is None:
            logger.error("Failed to load data from Hopsworks. Exiting.")
            return
        
        # Prepare features and target
        logger.info("Preparing features and target...")
        feature_cols = [col for col in df.columns if col not in ['date', 'start_station_name', 'trip_count', 'avg_duration']]
        target = 'trip_count'  # Predict trip count
        
        # Perform time series train/test split
        train_df, test_df = time_series_train_test_split(df, test_size=0.2)
        
        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_test = test_df[feature_cols]
        y_test = test_df[target]
        
        # Train model
        model, mae, model_path = train_model(X_train, y_train, X_test, y_test)
        
        # Register model in Hopsworks
        model_name = "citibike_trip_predictor"
        registered_model = register_model(project, model_path, model_name, mae)
        
        if registered_model:
            # Try to deploy the model
            deploy_success = deploy_model(project, model_name)
            
            if deploy_success:
                logger.info("Model training and deployment completed successfully")
            else:
                logger.warning("Model registered but deployment failed")
        else:
            logger.error("Failed to register model")
        
        logger.info("Model training pipeline completed")
    
    except Exception as e:
        logger.exception(f"Error in model training pipeline: {e}")

if __name__ == "__main__":
    main()
