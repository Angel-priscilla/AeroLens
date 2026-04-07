import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AeroLens-Flight Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">✈️ AeroLens-Flight Fare Prediction System</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Training", "Data Analysis"])

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    airlines = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara', 'GoAir']
    sources = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad']
    destinations = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad']
    
    data = {
        'Airline': np.random.choice(airlines, n_samples),
        'Source': np.random.choice(sources, n_samples),
        'Destination': np.random.choice(destinations, n_samples),
        'Total_Stops': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Duration_Hours': np.random.uniform(1, 24, n_samples),
        'Days_Left': np.random.randint(1, 365, n_samples),
        'Dep_Hour': np.random.randint(0, 24, n_samples),
        'Arrival_Hour': np.random.randint(0, 24, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Remove same source and destination
    df = df[df['Source'] != df['Destination']]
    
    # Create price based on features (realistic pricing logic)
    base_price = 3000
    
    # Airline premium
    airline_premium = {'IndiGo': 0, 'Air India': 500, 'Jet Airways': 1000, 
                      'SpiceJet': -200, 'Vistara': 800, 'GoAir': -300}
    
    # Route premium (distance factor)
    route_premium = np.random.uniform(0, 2000, len(df))
    
    # Stops penalty
    stops_penalty = df['Total_Stops'] * 500
    
    # Duration factor
    duration_factor = df['Duration_Hours'] * 100
    
    # Days left factor (booking in advance)
    days_factor = np.where(df['Days_Left'] < 7, 2000,  # Last minute booking
                          np.where(df['Days_Left'] < 30, 500, 0))  # Short notice
    
    # Time of day factor
    peak_hours = [6, 7, 8, 18, 19, 20]
    time_factor = np.where(df['Dep_Hour'].isin(peak_hours), 300, 0)
    
    df['Price'] = (base_price + 
                   df['Airline'].map(airline_premium) + 
                   route_premium + 
                   stops_penalty + 
                   duration_factor + 
                   days_factor + 
                   time_factor + 
                   np.random.normal(0, 500, len(df)))
    
    df['Price'] = np.maximum(df['Price'], 1000)  # Minimum price
    df['Price'] = df['Price'].round(0).astype(int)
    
    return df.reset_index(drop=True)

# Load data
df = generate_sample_data()

# Preprocessing function
def preprocess_data(data):
    le_airline = LabelEncoder()
    le_source = LabelEncoder()
    le_destination = LabelEncoder()
    
    data_processed = data.copy()
    data_processed['Airline_Encoded'] = le_airline.fit_transform(data['Airline'])
    data_processed['Source_Encoded'] = le_source.fit_transform(data['Source'])
    data_processed['Destination_Encoded'] = le_destination.fit_transform(data['Destination'])
    
    return data_processed, le_airline, le_source, le_destination

# Train model
@st.cache_data
def train_model():
    data_processed, le_airline, le_source, le_destination = preprocess_data(df)
    
    features = ['Airline_Encoded', 'Source_Encoded', 'Destination_Encoded', 
                'Total_Stops', 'Duration_Hours', 'Days_Left', 'Dep_Hour', 'Arrival_Hour']
    
    X = data_processed[features]
    y = data_processed['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, le_airline, le_source, le_destination, mae, r2, X_test, y_test, y_pred

# Get trained model
model, le_airline, le_source, le_destination, mae, r2, X_test, y_test, y_pred = train_model()

# Page content based on selection
if page == "Prediction":
    st.markdown("## 🎯 Predict Flight Fare")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Flight Details")
        
        airline = st.selectbox("Select Airline", df['Airline'].unique())
        source = st.selectbox("From", df['Source'].unique())
        destination = st.selectbox("To", df['Destination'].unique())
        
        if source == destination:
            st.error("Source and destination cannot be the same!")
        
        total_stops = st.selectbox("Total Stops", [0, 1, 2, 3])
        duration = st.slider("Flight Duration (hours)", 1.0, 24.0, 5.0, 0.5)
    
    with col2:
        st.markdown("### Booking Details")
        
        days_left = st.slider("Days left for departure", 1, 365, 30)
        dep_hour = st.slider("Departure Hour", 0, 23, 10)
        arrival_hour = st.slider("Arrival Hour", 0, 23, 15)
        
        if st.button("🔮 Predict Fare", type="primary"):
            if source != destination:
                # Prepare input data
                input_data = pd.DataFrame({
                    'Airline_Encoded': [le_airline.transform([airline])[0]],
                    'Source_Encoded': [le_source.transform([source])[0]],
                    'Destination_Encoded': [le_destination.transform([destination])[0]],
                    'Total_Stops': [total_stops],
                    'Duration_Hours': [duration],
                    'Days_Left': [days_left],
                    'Dep_Hour': [dep_hour],
                    'Arrival_Hour': [arrival_hour]
                })
                
                # Make prediction
                predicted_price = model.predict(input_data)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    Predicted Flight Fare: ₹{predicted_price:,.0f}
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 💡 Insights")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    if days_left < 7:
                        st.warning("⚠️ Last-minute booking - prices may be higher!")
                    elif days_left > 60:
                        st.success("✅ Good time to book - advance booking discount!")
                    
                    if total_stops == 0:
                        st.info("✈️ Direct flight - premium pricing")
                    elif total_stops >= 2:
                        st.info("🔄 Multiple stops - budget option")
                
                with col4:
                    peak_hours = [6, 7, 8, 18, 19, 20]
                    if dep_hour in peak_hours:
                        st.warning("⏰ Peak hour departure - higher prices")
                    else:
                        st.success("🕐 Off-peak departure - better rates")

elif page == "Model Training":
    st.markdown("## 🤖 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Mean Absolute Error</h3>
            <h2>₹{mae:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>R² Score</h3>
            <h2>{r2:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Model Accuracy</h3>
            <h2>{r2*100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("### 📊 Feature Importance")
    
    feature_names = ['Airline', 'Source', 'Destination', 'Total_Stops', 
                    'Duration_Hours', 'Days_Left', 'Dep_Hour', 'Arrival_Hour']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title="Feature Importance in Flight Fare Prediction",
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Prediction vs Actual
    st.markdown("### 🎯 Prediction vs Actual")
    
    comparison_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    fig_scatter = px.scatter(
        comparison_df, 
        x='Actual', 
        y='Predicted',
        title="Actual vs Predicted Flight Fares",
        labels={'Actual': 'Actual Price (₹)', 'Predicted': 'Predicted Price (₹)'}
    )
    
    # Add perfect prediction line
    min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
    max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "Data Analysis":
    st.markdown("## 📈 Data Analysis & Insights")
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset Overview")
        st.write(f"Total Records: {len(df):,}")
        st.write(f"Average Price: ₹{df['Price'].mean():,.0f}")
        st.write(f"Price Range: ₹{df['Price'].min():,} - ₹{df['Price'].max():,}")
    
    with col2:
        st.markdown("### Quick Stats")
        st.write(df[['Price', 'Duration_Hours', 'Days_Left', 'Total_Stops']].describe())
    
    # Price distribution
    st.markdown("### 💰 Price Distribution")
    
    fig_hist = px.histogram(
        df, 
        x='Price', 
        nbins=50,
        title="Distribution of Flight Prices",
        labels={'Price': 'Price (₹)', 'count': 'Frequency'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Price by airline
    st.markdown("### ✈️ Price by Airline")
    
    fig_box = px.box(
        df, 
        x='Airline', 
        y='Price',
        title="Price Distribution by Airline"
    )
    fig_box.update_xaxes(tickangle=45)
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### 🔥 Feature Correlations")
    
    corr_features = ['Price', 'Total_Stops', 'Duration_Hours', 'Days_Left', 'Dep_Hour', 'Arrival_Hour']
    corr_matrix = df[corr_features].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Route analysis
    st.markdown("### 🗺️ Popular Routes")
    
    route_analysis = df.groupby(['Source', 'Destination']).agg({
        'Price': ['mean', 'count']
    }).round(0)
    
    route_analysis.columns = ['Average_Price', 'Flight_Count']
    route_analysis = route_analysis.reset_index()
    route_analysis['Route'] = route_analysis['Source'] + ' → ' + route_analysis['Destination']
    route_analysis = route_analysis.sort_values('Flight_Count', ascending=False).head(10)
    
    fig_routes = px.bar(
        route_analysis,
        x='Route',
        y='Flight_Count',
        color='Average_Price',
        title="Top 10 Routes by Flight Count",
        labels={'Flight_Count': 'Number of Flights', 'Average_Price': 'Avg Price (₹)'}
    )
    fig_routes.update_xaxes(tickangle=45)
    st.plotly_chart(fig_routes, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit & Machine Learning</p>
    <p>Flight Fare Prediction System - Helping you find the best deals! ✈️</p>
</div>
""", unsafe_allow_html=True)