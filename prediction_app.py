import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# Try to import hopsworks, but handle it gracefully if it fails
try:
    import hopsworks
    import joblib
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="CitiBike Trip Prediction",
    layout="wide"
)

# App title and description
st.title("CitiBike Trip Prediction App")
st.markdown("""
This app predicts the number of trips for the top 3 CitiBike stations in New York City.
""")

# Function to create sample data
def create_sample_data():
    # Create a sample dataset with three stations
    stations = ["8 Ave & W 31 St", "Lafayette St & E 8 St", "Pershing Square North"]
    dates = pd.date_range(end=datetime.datetime.now().date(), periods=60)
    
    data = []
    for station in stations:
        for date in dates:
            # Generate some realistic data with weekly patterns
            base = 200 if station == "8 Ave & W 31 St" else 150 if station == "Lafayette St & E 8 St" else 120
            weekend_factor = 0.8 if date.dayofweek >= 5 else 1.2
            season_factor = 1.0 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)  # Seasonal pattern
            noise = np.random.normal(1.0, 0.15)  # Random noise
            
            trip_count = int(base * weekend_factor * season_factor * noise)
            
            data.append({
                "start_station_name": station,
                "date": date,
                "trip_count": trip_count,
                "is_weekend": 1 if date.dayofweek >= 5 else 0,
                "day_of_week": date.dayofweek,
                "month": date.month
            })
    
    return pd.DataFrame(data)

# Load data - with fallback to sample data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data with graceful fallback to sample data"""
    
    # If Hopsworks is not available, use sample data
    if not HOPSWORKS_AVAILABLE:
        st.warning("Hopsworks package not available. Using sample data.")
        return create_sample_data()
    
    try:
        # Try to connect to Hopsworks
        project = hopsworks.login(
            project="CitiBike_Final",
            api_key_value="NoSnqjvqruam2G2e.of4xXCy3fxpjkmgdpJflgRoTRbWkTsXdTM3hlQMGlyU37sXiqgLgGbSyBh57edxq"
        )
        fs = project.get_feature_store()
        
        try:
            # Try to get feature group
            fg = fs.get_feature_group(name="citibike_ts_features", version=1)
            
            # Use a different method to get data to avoid the 'select_all' issue
            query = fg.select(["start_station_name", "date", "trip_count", "is_weekend", "day_of_week", "month"])
            df = query.read()
            st.success("Successfully loaded data from Hopsworks Feature Store")
            return df
            
        except Exception as e:
            st.error(f"Error accessing feature group: {e}")
            return create_sample_data()
    
    except Exception as e:
        st.error(f"Error connecting to Hopsworks: {e}")
        return create_sample_data()

# Load our data
with st.spinner("Loading CitiBike data..."):
    df = load_data()

# Display basic info about the dataset
st.subheader("Dataset Overview")
stations = sorted(df['start_station_name'].unique())
st.write(f"Number of stations: {len(stations)}")

# Convert date column to datetime if it's not already
if not pd.api.types.is_datetime64_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'])

st.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Station selection
selected_station = st.selectbox("Select a station:", stations)

# Filter data for the selected station
station_data = df[df['start_station_name'] == selected_station].sort_values('date')

# Display historical data
st.subheader(f"Historical Trip Data for {selected_station}")

# Create a chart of historical data
fig = px.line(station_data, x='date', y='trip_count', 
              title=f"Trip Count for {selected_station}",
              labels={'date': 'Date', 'trip_count': 'Number of Trips'})
fig.update_layout(xaxis_title='Date', yaxis_title='Number of Trips')
st.plotly_chart(fig, use_container_width=True)

# Historical patterns
st.subheader("Historical Patterns")

# Day of week pattern
dow_data = station_data.groupby('day_of_week')['trip_count'].mean().reset_index()
dow_data['day_name'] = dow_data['day_of_week'].map({
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
})

fig = px.bar(dow_data, x='day_name', y='trip_count', 
             title="Average Trips by Day of Week",
             labels={'day_name': 'Day', 'trip_count': 'Average Trips'})
st.plotly_chart(fig, use_container_width=True)

# Monthly pattern
if len(station_data['month'].unique()) > 1:
    month_data = station_data.groupby('month')['trip_count'].mean().reset_index()
    month_data['month_name'] = month_data['month'].map({
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    })
    
    fig = px.bar(month_data, x='month_name', y='trip_count', 
                title="Average Trips by Month",
                labels={'month_name': 'Month', 'trip_count': 'Average Trips'})
    st.plotly_chart(fig, use_container_width=True)

# Prediction functionality
st.subheader("Trip Prediction")

# Date range for prediction
forecast_days = st.slider("Number of days to forecast:", 1, 30, 7)
start_date = datetime.datetime.now().date() + datetime.timedelta(days=1)
forecast_dates = [start_date + datetime.timedelta(days=i) for i in range(forecast_days)]

# Model to generate predictions
def generate_predictions(station_data, forecast_dates):
    """Generate predictions based on historical patterns"""
    
    # Calculate average trips by day of week
    dow_averages = station_data.groupby('day_of_week')['trip_count'].mean().to_dict()
    
    # Calculate monthly factors if available
    monthly_factors = {}
    if len(station_data['month'].unique()) > 3:  # Only use monthly patterns if we have enough data
        monthly_avg = station_data['trip_count'].mean()
        for month, group in station_data.groupby('month'):
            monthly_factors[month] = group['trip_count'].mean() / monthly_avg
    
    # Make predictions
    predictions = []
    for date in forecast_dates:
        day_of_week = date.weekday()
        month = date.month
        
        # Base prediction on day of week average
        base_prediction = dow_averages.get(day_of_week, station_data['trip_count'].mean())
        
        # Apply monthly factor if available
        if month in monthly_factors:
            base_prediction *= monthly_factors[month]
        
        # Add some randomness
        prediction = max(0, int(base_prediction * (0.9 + 0.2 * np.random.random())))
        
        predictions.append({
            'date': date,
            'prediction': prediction,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'month': month
        })
    
    return pd.DataFrame(predictions)

# Generate predictions
with st.spinner("Generating predictions..."):
    predictions_df = generate_predictions(station_data, forecast_dates)

# Display predictions
st.subheader(f"Predicted Trips for {selected_station}")

st.dataframe(predictions_df[['date', 'prediction', 'day_of_week', 'is_weekend']])

# Create chart with predictions
last_30_days = station_data[station_data['date'] >= (datetime.datetime.now().date() - datetime.timedelta(days=30))]

fig = go.Figure()
# Historical data
fig.add_trace(go.Scatter(
    x=last_30_days['date'], 
    y=last_30_days['trip_count'],
    mode='lines+markers',
    name='Historical',
    line=dict(color='blue')
))
# Predictions
fig.add_trace(go.Scatter(
    x=predictions_df['date'], 
    y=predictions_df['prediction'],
    mode='lines+markers',
    name='Predicted',
    line=dict(color='red')
))

fig.update_layout(
    title=f"Trip Prediction for {selected_station}",
    xaxis_title='Date',
    yaxis_title='Number of Trips',
    legend=dict(x=0, y=1, traceorder='normal'),
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# Download predictions
st.download_button(
    label="Download Predictions as CSV",
    data=predictions_df.to_csv(index=False).encode('utf-8'),
    file_name=f"citibike_predictions_{selected_station.replace(' ', '_')}.csv",
    mime='text/csv',
)

