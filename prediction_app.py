import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page config
st.set_page_config(
    page_title="CitiBike Trip Prediction",
    page_icon="🚲",
    layout="wide"
)

# App title and description
st.title("CitiBike Trip Prediction App")
st.markdown("""
This app predicts the number of trips for the top 3 CitiBike stations in New York City.
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

# Function to create sample data if Hopsworks connection fails
def create_sample_data():
    # Create a sample dataset with three stations
    stations = ["Station A", "Station B", "Station C"]
    dates = pd.date_range(end=datetime.datetime.now().date(), periods=30)
    
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

# Load our data
with st.spinner("Loading CitiBike data..."):
    df = load_feature_data()

# Display basic info about the dataset
st.subheader("Dataset Overview")
stations = df['start_station_name'].unique()
st.write(f"Number of stations: {len(stations)}")
st.write(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Station selection
selected_station = st.selectbox("Select a station:", stations)

# Filter data for the selected station
station_data = df[df['start_station_name'] == selected_station].sort_values('date')

# Display historical data
st.subheader(f"Historical Trip Data for {selected_station}")

# Create a chart of historical data
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='date', y='trip_count', data=station_data, ax=ax)
ax.set_title(f"Trip Count for {selected_station}")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Trips")
ax.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Prediction functionality
st.subheader("Trip Prediction")

# Date range for prediction
forecast_days = st.slider("Number of days to forecast:", 1, 30, 7)
start_date = datetime.datetime.now().date() + datetime.timedelta(days=1)
forecast_dates = [start_date + datetime.timedelta(days=i) for i in range(forecast_days)]

# Simple model to generate predictions
def generate_predictions(station_data, forecast_dates):
    # This is a simplified prediction model
    # In a real scenario, we would load the trained model and use it
    
    # Calculate average trips by day of week
    dow_averages = station_data.groupby('day_of_week')['trip_count'].mean().to_dict()
    
    # Make predictions
    predictions = []
    for date in forecast_dates:
        day_of_week = date.weekday()
        
        # Base prediction on day of week average
        base_prediction = dow_averages.get(day_of_week, station_data['trip_count'].mean())
        
        # Add some randomness
        prediction = max(0, int(base_prediction * (0.9 + 0.2 * np.random.random())))
        
        predictions.append({
            'date': date,
            'prediction': prediction,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0
        })
    
    return pd.DataFrame(predictions)

# Generate predictions
with st.spinner("Generating predictions..."):
    predictions_df = generate_predictions(station_data, forecast_dates)

# Display predictions
st.subheader(f"Predicted Trips for {selected_station}")
st.dataframe(predictions_df[['date', 'prediction', 'day_of_week', 'is_weekend']])

# Create chart with predictions
fig, ax = plt.subplots(figsize=(10, 6))
# Historical data
sns.lineplot(x='date', y='trip_count', data=station_data.tail(30), ax=ax, label='Historical')
# Predictions
sns.lineplot(x='date', y='prediction', data=predictions_df, ax=ax, color='red', label='Predicted')

ax.set_title(f"Trip Prediction for {selected_station}")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Trips")
ax.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Download predictions
st.download_button(
    label="Download Predictions as CSV",
    data=predictions_df.to_csv(index=False).encode('utf-8'),
    file_name=f"citibike_predictions_{selected_station.replace(' ', '_')}.csv",
    mime='text/csv',
)


