#!/usr/bin/env python
"""
Citibike Model Training Pipeline
This script retrieves feature data from Hopsworks, trains models, and registers the best model.
"""

import pandas as pd
import numpy as np
import hopsworks
import mlflow
import joblib
import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
import lightgbm as lgbm
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns

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

def setup_mlflow():
    """
    Set up MLflow tracking.
    """
    logger.info("Setting up MLflow...")
    
    # Try to use the MLflow tracking URI from environment variable
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Using MLflow tracking URI from environment: {tracking_uri}")
    else:
        # Set up local tracking as fallback
        os.makedirs('mlruns', exist_ok=True)
        mlflow.set_tracking_uri("file:./mlruns")
        logger.info("Using local directory for MLflow tracking: ./mlruns")
    
    # Set the experiment
    mlflow.set_experiment("CitiBike_Production")
    
    # Return the tracking URI for later use
    return mlflow.get_tracking_uri()

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
    logger.info(f"Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
    logger.info(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
    
    return train_df, test_df

def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a baseline model (mean predictor).
    """
    logger.info("Training baseline model...")
    
    # Start MLflow run
    with mlflow.start_run(run_name="baseline_model") as run:
        # Log parameters
        mlflow.log_param("model_type", "DummyRegressor")
        mlflow.log_param("strategy", "mean")
        
        # Train model
        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        logger.info(f"Baseline model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        # Save model locally
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, "models/baseline_model.joblib")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "baseline_model")
        
        return model, mae, rmse, run.info.run_id

def train_full_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a full LightGBM model with all features.
    """
    logger.info("Training full LightGBM model...")
    
    # Start MLflow run
    with mlflow.start_run(run_name="full_lgbm_model") as run:
        # Set parameters
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "num_leaves": 31,
            "random_state": 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LightGBM")
        
        # Train model
        model = lgbm.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        logger.info(f"Full LightGBM model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Feature Importance (Top 20)')
        plt.tight_layout()
        
        # Save plot locally
        plt.savefig("feature_importance.png")
        
        # Log plot to MLflow
        mlflow.log_artifact("feature_importance.png")
        
        # Save model locally
        joblib.dump(model, "models/full_lgbm_model.joblib")
        
        # Log model to MLflow
        mlflow.lightgbm.log_model(model, "full_lgbm_model")
        
        return model, mae, rmse, feature_importance, run.info.run_id

def train_optimized_model(X_train, y_train, X_test, y_test, feature_importance=None):
    """
    Train and evaluate an optimized LightGBM model with feature selection and tuned parameters.
    """
    logger.info("Training optimized LightGBM model...")
    
    # Start MLflow run
    with mlflow.start_run(run_name="optimized_lgbm_model") as run:
        # Select top 10 features if feature importance is provided
        if feature_importance is not None:
            top_features = feature_importance['feature'].head(10).tolist()
            X_train_reduced = X_train[top_features]
            X_test_reduced = X_test[top_features]
            
            # Log the selected features
            mlflow.log_param("feature_selection", "importance_based")
            mlflow.log_param("num_features", len(top_features))
            mlflow.log_param("selected_features", ", ".join(top_features))
        else:
            # Use SelectKBest if no feature importance provided
            selector = SelectKBest(f_regression, k=10)
            X_train_reduced = selector.fit_transform(X_train, y_train)
            X_test_reduced = selector.transform(X_test)
            
            # Log parameters
            mlflow.log_param("feature_selection", "f_regression")
            mlflow.log_param("num_features", 10)
        
        # Optimized parameters (tuned for better performance)
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 7,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LightGBM_Optimized")
        
        # Train model
        model = lgbm.LGBMRegressor(**params)
        
        if feature_importance is not None:
            model.fit(X_train_reduced, y_train)
            y_pred = model.predict(X_test_reduced)
        else:
            model.fit(X_train_reduced, y_train)
            y_pred = model.predict(X_test_reduced)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        logger.info(f"Optimized LightGBM model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        # Save model locally
        joblib.dump(model, "models/optimized_lgbm_model.joblib")
        
        # Log model to MLflow
        mlflow.lightgbm.log_model(model, "optimized_lgbm_model")
        
        # Save feature info for inference
        if feature_importance is not None:
            feature_info = {
                "features": top_features,
                "selection_method": "importance_based"
            }
            joblib.dump(feature_info, "models/feature_info.joblib")
            mlflow.log_artifact("models/feature_info.joblib")
        
        return model, mae, rmse, run.info.run_id

def register_best_model(project, best_run_id, model_name, mae):
    """
    Register the best model in Hopsworks Model Registry.
    """
    logger.info(f"Registering best model (run_id: {best_run_id}) to Hopsworks Model Registry...")
    
    try:
        # Get model registry
        mr = project.get_model_registry()
        
        # Create model in registry
        model = mr.get_or_create_model(
            name=model_name,
            version=1,
            description=f"CitiBike trip prediction model (MAE: {mae:.2f})",
            metrics={"mae": mae}
        )
        
        # The model path from MLflow
        model_path = f"mlruns/1/{best_run_id}/artifacts/optimized_lgbm_model"
        
        # Save locally trained model to the model registry
        model.save(model_path)
        
        logger.info(f"Successfully registered model: {model_name}")
        return True
    
    except Exception as e:
        logger.exception(f"Error registering model: {e}")
        return False

def create_model_comparison_visualization(baseline_mae, full_mae, optimized_mae):
    """
    Create visualization comparing the models.
    """
    logger.info("Creating model comparison visualization...")
    
    try:
        # Prepare data for visualization
        models = ['Baseline', 'Full LightGBM', 'Optimized LightGBM']
        maes = [baseline_mae, full_mae, optimized_mae]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, maes, color=['#ff9999', '#66b3ff', '#99ff99'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Model Comparison - Mean Absolute Error (MAE)')
        plt.xlabel('Model')
        plt.ylabel('MAE (lower is better)')
        plt.ylim(0, max(maes) * 1.2)  # Add some headroom
        plt.tight_layout()
        
        # Save the figure
        plt.savefig("model_comparison.png")
        
        # Log to MLflow
        with mlflow.start_run(run_name="model_comparison"):
            mlflow.log_metric("baseline_mae", baseline_mae)
            mlflow.log_metric("full_model_mae", full_mae)
            mlflow.log_metric("optimized_model_mae", optimized_mae)
            mlflow.log_artifact("model_comparison.png")
            
        logger.info("Model comparison visualization created and logged")
        return True
    
    except Exception as e:
        logger.exception(f"Error creating model comparison visualization: {e}")
        return False

def deploy_model_to_hopsworks(project, model_name):
    """
    Deploy the model to Hopsworks serving.
    """
    logger.info(f"Deploying model {model_name} to Hopsworks serving...")
    
    try:
        # Get model registry
        mr = project.get_model_registry()
        
        # Get the model
        model = mr.get_model(name=model_name, version=1)
        
        # Create a serving instance if it doesn't exist
        serving = project.get_model_serving()
        
        # Create the deployment
        deployment = serving.deploy(model_name, model=model)
        
        # Return the deployment status
        logger.info(f"Model deployed successfully: {model_name}")
        return True
    
    except Exception as e:
        logger.exception(f"Error deploying model: {e}")
        return False

def main():
    """
    Main function to execute the model training pipeline.
    """
    logger.info("Starting CitiBike model training pipeline")
    
    try:
        # Set up MLflow
        tracking_uri = setup_mlflow()
        
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
        
        logger.info(f"Features: {len(feature_cols)} columns")
        
        # Train baseline model
        baseline_model, baseline_mae, baseline_rmse, baseline_run_id = train_baseline_model(X_train, y_train, X_test, y_test)
        
        # Train full model
        full_model, full_mae, full_rmse, feature_importance, full_run_id = train_full_model(X_train, y_train, X_test, y_test)
        
        # Train optimized model
        optimized_model, optimized_mae, optimized_rmse, optimized_run_id = train_optimized_model(X_train, y_train, X_test, y_test, feature_importance)
        
        # Create model comparison visualization
        create_model_comparison_visualization(baseline_mae, full_mae, optimized_mae)
        
        # Determine the best model
        mae_values = {
            "baseline": (baseline_mae, baseline_run_id),
            "full": (full_mae, full_run_id),
            "optimized": (optimized_mae, optimized_run_id)
        }
        
        best_model_name = min(mae_values, key=lambda k: mae_values[k][0])
        best_mae, best_run_id = mae_values[best_model_name]
        
        logger.info(f"Best model: {best_model_name} (MAE: {best_mae:.2f})")
        
        # For this pipeline, we'll always register the optimized model
        # regardless of which performed best, as it's typically more efficient
        model_name = "citibike_trip_predictor"
        register_success = register_best_model(project, optimized_run_id, model_name, optimized_mae)
        
        if register_success:
            # Deploy model to Hopsworks serving
            deploy_success = deploy_model_to_hopsworks(project, model_name)
            
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