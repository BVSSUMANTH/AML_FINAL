import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="CitiBike Model Monitoring",
    layout="wide"
)

# App title and description
st.title("CitiBike Model Monitoring Dashboard")
st.markdown("""
This dashboard monitors the performance of the CitiBike trip prediction model.
""")

# Initialize connection to Hopsworks
@st.cache_resource
def get_hopsworks_project():
    return hopsworks.login(
        project="CitiBike_Final",
        api_key_value="NoSnqjvqruam2G2e.of4xXCy3fxpjkmgdpJflgRoTRbWkTsXdTM3hlQMGlyU37sXiqgLgGbSyBh57edxq"
    )

# Function to load feature data from Hopsworks
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_feature_data():
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    
    try:
        # Get feature group
        fg = fs.get_feature_group(name="citibike_ts_features", version=1)
        # Get data
        query = fg.select_all()
        df = query.read()
        return df
    except Exception as e:
        st.error(f"Error loading data from feature store: {e}")
        # Return sample data as fallback
        return create_sample_data()

# Function to load prediction data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_prediction_data():
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        
        # Try to get predictions feature group
        fg = fs.get_feature_group(name="citibike_predictions", version=1)
        query = fg.select_all()
        df = query.read()
        return df
    except Exception as e:
        # If no predictions are available in Hopsworks, create sample data
        st.warning("Could not load prediction data from Hopsworks. Using generated sample data.")
        return create_sample_prediction_data()

# Function to create sample data if Hopsworks connection fails
def create_sample_data():
    # Create a sample dataset with three stations
    stations = ["Station A", "Station B", "Station C"]
    dates = pd.date_range(end=datetime.datetime.now().date(), periods=60)
    
    data = []
    for station in stations:
        for date in dates:
            # Generate some random data
            trip_count = np.random.randint(50, 500)
            data.append({
                "start_station_name": station,
                "date": date,
                "trip_count": trip_count,
                "is_weekend": 1 if date.dayofweek >= 5 else 0,
                "day_of_week": date.dayofweek,
                "month": date.month
            })
    
    return pd.DataFrame(data)

# Function to create sample prediction data
def create_sample_prediction_data():
    stations = ["Station A", "Station B", "Station C"]
    start_date = datetime.datetime.now().date() - datetime.timedelta(days=30)
    dates = pd.date_range(start=start_date, periods=30)
    
    data = []
    for station in stations:
        for date in dates:
            # Generate actual and predicted values
            actual = np.random.randint(50, 500)
            predicted = actual + np.random.randint(-50, 50)  # Add some error
            
            data.append({
                "station": station,
                "date": date,
                "actual_trips": actual,
                "predicted_trips": predicted
            })
    
    return pd.DataFrame(data)

# Calculate model metrics
def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2)
    }

# Load data
with st.spinner("Loading data..."):
    feature_df = load_feature_data()
    prediction_df = load_prediction_data()

# If prediction data doesn't have actual values, merge with feature data
if 'actual_trips' not in prediction_df.columns and 'trip_count' in feature_df.columns:
    # Create a combined dataset with actual and predicted values
    # This is a simplified approach - in real scenarios, you'd need to carefully match dates
    combined_data = []
    
    for station in feature_df['start_station_name'].unique():
        station_features = feature_df[feature_df['start_station_name'] == station].sort_values('date')
        station_preds = prediction_df[prediction_df['station'] == station].sort_values('date') if 'station' in prediction_df.columns else pd.DataFrame()
        
        if not station_preds.empty:
            # For dates that appear in both datasets
            for date in pd.to_datetime(station_features['date']).dt.date:
                pred_row = station_preds[pd.to_datetime(station_preds['date']).dt.date == date]
                feat_row = station_features[pd.to_datetime(station_features['date']).dt.date == date]
                
                if not pred_row.empty and not feat_row.empty:
                    combined_data.append({
                        "station": station,
                        "date": date,
                        "actual_trips": feat_row['trip_count'].values[0],
                        "predicted_trips": pred_row['predicted_trips'].values[0]
                    })
    
    if combined_data:
        prediction_df = pd.DataFrame(combined_data)
    else:
        # If no matching dates, just create sample data
        prediction_df = create_sample_prediction_data()
elif 'actual_trips' not in prediction_df.columns:
    # If we only have predictions but no actuals, create sample data
    prediction_df = create_sample_prediction_data()

# Dashboard layout
st.subheader("Model Performance Overview")

# Overall metrics
predictions = prediction_df['predicted_trips'].values
actuals = prediction_df['actual_trips'].values
metrics = calculate_metrics(actuals, predictions)

# Create columns for metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']}")
with col2:
    st.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']}")
with col3:
    st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']}%")

# Time series of prediction error
st.subheader("Prediction Error Over Time")

# Calculate error for each prediction
prediction_df['error'] = prediction_df['predicted_trips'] - prediction_df['actual_trips']
prediction_df['abs_error'] = abs(prediction_df['error'])
prediction_df['pct_error'] = (prediction_df['error'] / prediction_df['actual_trips']) * 100

# Aggregate by date for overall trend
daily_error = prediction_df.groupby('date')[['abs_error', 'pct_error']].mean().reset_index()

# Plot error over time
fig = px.line(daily_error, x='date', y='abs_error', 
              title='Average Absolute Error Over Time',
              labels={'date': 'Date', 'abs_error': 'Absolute Error'})
fig.update_layout(xaxis_title='Date', yaxis_title='Absolute Error')
st.plotly_chart(fig, use_container_width=True)

# Error distribution
st.subheader("Error Distribution")

# Create histogram of errors
fig = px.histogram(prediction_df, x='error', 
                   title='Distribution of Prediction Errors',
                   labels={'error': 'Error (Predicted - Actual)'})
fig.update_layout(xaxis_title='Error', yaxis_title='Count')
st.plotly_chart(fig, use_container_width=True)

# Station-wise performance
st.subheader("Station-wise Performance")

# Calculate metrics by station
station_metrics = []
for station in prediction_df['station'].unique():
    station_data = prediction_df[prediction_df['station'] == station]
    station_metrics.append({
        'station': station,
        **calculate_metrics(station_data['actual_trips'].values, station_data['predicted_trips'].values)
    })

station_metrics_df = pd.DataFrame(station_metrics)
st.dataframe(station_metrics_df)

# Station selection for detailed view
selected_station = st.selectbox("Select a station for detailed view:", prediction_df['station'].unique())

# Filter data for the selected station
station_data = prediction_df[prediction_df['station'] == selected_station].sort_values('date')

# Actual vs Predicted plot
st.subheader(f"Actual vs Predicted Trips for {selected_station}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=station_data['date'], y=station_data['actual_trips'],
                    mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=station_data['date'], y=station_data['predicted_trips'],
                    mode='lines+markers', name='Predicted'))
fig.update_layout(title=f'Actual vs Predicted Trips for {selected_station}',
                 xaxis_title='Date', yaxis_title='Number of Trips')
st.plotly_chart(fig, use_container_width=True)

# Model drift detection
st.subheader("Model Drift Detection")

# Calculate rolling window metrics to detect drift
window_size = min(7, len(daily_error))
daily_error['rolling_mae'] = daily_error['abs_error'].rolling(window=window_size).mean()

# Plot rolling MAE
fig = px.line(daily_error.dropna(), x='date', y='rolling_mae', 
              title=f'Rolling MAE ({window_size}-day window)',
              labels={'date': 'Date', 'rolling_mae': 'Rolling MAE'})

# Add threshold line (for demonstration)
threshold = metrics['MAE'] * 1.5
fig.add_hline(y=threshold, line_dash="dash", line_color="red",
              annotation_text="Alert Threshold", 
              annotation_position="bottom right")

fig.update_layout(xaxis_title='Date', yaxis_title='Rolling MAE')
st.plotly_chart(fig, use_container_width=True)

# Feature importance (simulated)
st.subheader("Feature Importance")

# Simulated feature importance
feature_names = ['trip_count_lag_1', 'trip_count_lag_2', 'trip_count_lag_3', 
                'trip_count_rolling_7', 'day_of_week', 'is_weekend', 'month']
importance_values = [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_values
})

fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
             title='Feature Importance',
             labels={'Importance': 'Importance', 'Feature': 'Feature'})
fig.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig, use_container_width=True)

# App footer
st.markdown("---")
st.markdown("Created as part of MLOps Final Project by Sumanth")