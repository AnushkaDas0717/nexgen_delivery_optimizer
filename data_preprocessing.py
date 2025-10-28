"""
Data Preprocessing Module
Handles data loading, cleaning, merging, and feature engineering
"""

import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def load_datasets():
    """Load all CSV datasets"""
    try:
        orders = pd.read_csv('data/orders.csv')
        delivery = pd.read_csv('data/delivery_performance.csv')
        routes = pd.read_csv('data/routes_distance.csv')
        vehicles = pd.read_csv('data/vehicle_fleet.csv')
        costs = pd.read_csv('data/cost_breakdown.csv')
        feedback = pd.read_csv('data/customer_feedback.csv')
        
        return orders, delivery, routes, vehicles, costs, feedback
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.info("Please ensure all CSV files are in the same directory as the app.")
        return None, None, None, None, None, None


def clean_data(df):
    """Basic data cleaning"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    return df


@st.cache_data
def merge_datasets(orders, delivery, routes, costs, feedback):
    """Merge all datasets on Order_ID"""
    
    # Start with delivery as base (has performance metrics)
    merged = delivery.copy()
    
    # Merge with orders
    merged = merged.merge(orders, on='Order_ID', how='left')
    
    # Merge with routes
    merged = merged.merge(routes, on='Order_ID', how='left')
    
    # Merge with costs
    merged = merged.merge(costs, on='Order_ID', how='left')
    
    # Merge with feedback (optional, some orders may not have feedback)
    merged = merged.merge(feedback[['Order_ID', 'Rating', 'Issue_Category', 'Would_Recommend']], 
                         on='Order_ID', how='left', suffixes=('', '_feedback'))
    
    return merged


def create_features(df):
    """Create derived features for ML"""
    
    # Create binary delay flag
    df['Delayed'] = (df['Actual_Delivery_Days'] > df['Promised_Delivery_Days']).astype(int)
    
    # Delay in days
    df['Delay_Days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
    
    # Total delivery cost
    cost_columns = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 
                   'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead']
    df['Total_Cost'] = df[cost_columns].sum(axis=1)
    
    # Cost per km
    df['Cost_Per_KM'] = df['Total_Cost'] / (df['Distance_KM'] + 1)  # +1 to avoid division by zero
    
    # Encode categorical variables for ML
    df['Priority_Encoded'] = df['Priority'].map({'Express': 2, 'Standard': 1, 'Economy': 0})
    
    # Weather impact binary
    df['Has_Weather_Impact'] = (df['Weather_Impact'] != 'None').astype(int)
    
    # Quality issue binary
    df['Has_Quality_Issue'] = (df['Quality_Issue'] != 'Perfect').astype(int)
    
    # On-time delivery flag
    df['On_Time'] = (df['Delivery_Status'] == 'On-Time').astype(int)
    
    return df


def compute_kpis(df):
    """Compute key performance indicators"""
    
    kpis = {
        'total_orders': len(df),
        'on_time_percentage': (df['On_Time'].sum() / len(df) * 100),
        'avg_delay_days': df[df['Delayed'] == 1]['Delay_Days'].mean(),
        'avg_rating': df['Customer_Rating'].mean(),
        'avg_cost': df['Total_Cost'].mean(),
        'severe_delays': (df['Delivery_Status'] == 'Severely-Delayed').sum(),
        'avg_distance': df['Distance_KM'].mean(),
        'total_fuel_consumption': df['Fuel_Consumption_L'].sum()
    }
    
    return kpis


@st.cache_data
def preprocess_all_data():
    """Main preprocessing pipeline"""
    
    # Load datasets
    orders, delivery, routes, vehicles, costs, feedback = load_datasets()
    
    if orders is None:
        return None, None
    
    # Clean individual datasets
    orders = clean_data(orders)
    delivery = clean_data(delivery)
    routes = clean_data(routes)
    costs = clean_data(costs)
    feedback = clean_data(feedback)
    
    # Merge datasets
    merged_df = merge_datasets(orders, delivery, routes, costs, feedback)
    
    # Create features
    merged_df = create_features(merged_df)
    
    # Compute KPIs
    kpis = compute_kpis(merged_df)
    
    return merged_df, kpis


def get_feature_columns():
    """Return list of features for ML model"""
    return [
        'Distance_KM',
        'Priority_Encoded',
        'Traffic_Delay_Minutes',
        'Has_Weather_Impact',
        'Toll_Charges_INR',
        'Fuel_Consumption_L',
        'Order_Value_INR'
    ]


def prepare_ml_data(df):
    """Prepare data for machine learning"""
    
    feature_cols = get_feature_columns()
    
    # Filter to only rows with all required features
    ml_df = df[feature_cols + ['Delayed']].copy()
    ml_df = ml_df.dropna()
    
    X = ml_df[feature_cols]
    y = ml_df['Delayed']
    
    return X, y, feature_cols