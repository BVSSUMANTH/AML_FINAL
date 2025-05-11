#!/usr/bin/env python
"""
Basic Model Registry Example
This script focuses only on registering a model in Hopsworks.
"""

import hopsworks
import joblib
import os
import logging
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to register a model in Hopsworks.
    """
    logger.info("Starting basic model registration")
    
    try:
        # Create a dummy model
        logger.info("Creating a simple model...")
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Fit on small dummy data
        X = np.random.rand(100, 4)
        y = np.random.rand(100)
        model.fit(X, y)
        
        # Save model locally
        os.makedirs('models', exist_ok=True)
        model_path = "models/dummy_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Connect to Hopsworks
        logger.info("Connecting to Hopsworks...")
        project = hopsworks.login(
            project="CitiBike_Final",
            api_key_value="NoSnqjvqruam2G2e.of4xXCy3fxpjkmgdpJflgRoTRbWkTsXdTM3hlQMGlyU37sXiqgLgGbSyBh57edxq"
        )
        logger.info(f"Connected to project: {project.name}")
        
        # Get the model registry - print its type
        logger.info("Getting model registry...")
        mr = project.get_model_registry()
        logger.info(f"Model registry type: {type(mr)}")
        logger.info(f"Available methods: {[m for m in dir(mr) if not m.startswith('_')]}")
        
        # Try different approaches to register the model
        
        # Approach 1: Using sklearn module
        logger.info("Trying to register model with sklearn module...")
        try:
            if hasattr(mr, 'sklearn'):
                model_instance = mr.sklearn.create_model(
                    name="dummy_model_sklearn",
                    version=1,
                    metrics={"dummy_metric": 0.5},
                    description="A dummy model for testing"
                )
                model_instance.save(model_path)
                logger.info("Successfully registered model with sklearn module")
            else:
                logger.warning("sklearn module not available on model registry")
        except Exception as e:
            logger.exception(f"Error registering with sklearn module: {e}")
        
        # Approach 2: Using python module
        logger.info("Trying to register model with python module...")
        try:
            if hasattr(mr, 'python'):
                model_instance = mr.python.create_model(
                    name="dummy_model_python",
                    metrics={"dummy_metric": 0.5},
                    description="A dummy model for testing"
                )
                model_instance.save(model_path)
                logger.info("Successfully registered model with python module")
            else:
                logger.warning("python module not available on model registry")
        except Exception as e:
            logger.exception(f"Error registering with python module: {e}")
        
        # Approach 3: Using create_model directly on mr
        logger.info("Trying to register model directly with create_model...")
        try:
            model_instance = mr.create_model(
                name="dummy_model_direct",
                version=1,
                metrics={"dummy_metric": 0.5},
                description="A dummy model for testing"
            )
            model_instance.save(model_path)
            logger.info("Successfully registered model directly with create_model")
        except Exception as e:
            logger.exception(f"Error registering directly with create_model: {e}")
            
        # Approach 4: Get or create model
        logger.info("Trying to get_or_create_model...")
        try:
            if hasattr(mr, 'get_or_create_model'):
                model_instance = mr.get_or_create_model(
                    name="dummy_model_get_or_create",
                    version=1,
                    description="A dummy model for testing",
                    metrics={"dummy_metric": 0.5}
                )
                model_instance.save(model_path)
                logger.info("Successfully used get_or_create_model")
            else:
                logger.warning("get_or_create_model not available on model registry")
        except Exception as e:
            logger.exception(f"Error with get_or_create_model: {e}")
            
        logger.info("Model registration attempts completed")
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
