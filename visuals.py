"""
Visualization Module
Contains all plotting functions for the dashboard
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st


def plot_kpi_cards(kpis):
    """Display KPI metrics in a card layout"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="On-Time Delivery %",
            value=f"{kpis['on_time_percentage']:.1f}%",
            delta=f"{kpis['on_time_percentage'] - 70:.1f}%" if kpis['on_time_percentage'] > 70 else None
        )
    
    with col2:
        st.metric(
            label="Average Delay (days)",
            value=f"{kpis['avg_delay_days']:.2f}" if not pd.isna(kpis['avg_delay_days']) else "N/A",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Average Rating",
            value=f"{kpis['avg_rating']:.2f}",
            delta=f"{kpis['avg_rating'] - 3:.2f}"
        )
    
    with col4:
        st.metric(
            label="Avg Delivery Cost (₹)",
            value=f"₹{kpis['avg_cost']:.2f}",
            delta=None
        )


def plot_delay_by_priority(df):
    """Bar chart: Delay rate by priority"""
    
    delay_by_priority = df.groupby('Priority')['Delayed'].mean() * 100
    delay_by_priority = delay_by_priority.sort_values(ascending=False)
    
    fig = px.bar(
        x=delay_by_priority.index,
        y=delay_by_priority.values,
        labels={'x': 'Priority Level', 'y': 'Delay Rate (%)'},
        title='Delay Rate by Priority Level',
        color=delay_by_priority.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(showlegend=False, height=400)
    return fig


def plot_cost_vs_delay(df):
    """Scatter plot: Cost vs Delay relationship"""
    
    fig = px.scatter(
        df,
        x='Total_Cost',
        y='Delay_Days',
        color='Delivery_Status',
        size='Distance_KM',
        hover_data=['Order_ID', 'Carrier'],
        title='Delivery Cost vs Delay Days',
        labels={'Total_Cost': 'Total Cost (₹)', 'Delay_Days': 'Delay (days)'}
    )
    
    fig.update_layout(height=450)
    return fig


def plot_rating_distribution(df):
    """Histogram: Customer rating distribution"""
    
    fig = px.histogram(
        df,
        x='Customer_Rating',
        nbins=5,
        title='Customer Rating Distribution',
        labels={'Customer_Rating': 'Rating', 'count': 'Number of Orders'},
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(height=400)
    return fig


def plot_delivery_status_pie(df):
    """Pie chart: Delivery status breakdown"""
    
    status_counts = df['Delivery_Status'].value_counts()
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title='Delivery Status Distribution',
        hole=0.4
    )
    
    fig.update_layout(height=400)
    return fig


def plot_carrier_performance(df):
    """Bar chart: Carrier performance comparison"""
    
    carrier_perf = df.groupby('Carrier').agg({
        'On_Time': 'mean',
        'Customer_Rating': 'mean'
    }).reset_index()
    
    carrier_perf['On_Time'] = carrier_perf['On_Time'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='On-Time %',
        x=carrier_perf['Carrier'],
        y=carrier_perf['On_Time'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Avg Rating (x20)',
        x=carrier_perf['Carrier'],
        y=carrier_perf['Customer_Rating'] * 20,
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Carrier Performance Comparison',
        xaxis_title='Carrier',
        yaxis_title='Percentage / Scaled Rating',
        barmode='group',
        height=450
    )
    
    return fig


def plot_correlation_heatmap(df):
    """Correlation heatmap for numerical features"""
    
    numerical_cols = [
        'Distance_KM', 'Traffic_Delay_Minutes', 'Order_Value_INR',
        'Total_Cost', 'Delay_Days', 'Customer_Rating', 'Fuel_Consumption_L'
    ]
    
    # Filter columns that exist
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    corr_matrix = df[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, pad=20)
    
    return fig


def plot_feature_importance(feature_importance):
    """Bar chart: ML feature importance"""
    
    fig = px.bar(
        feature_importance.head(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Feature Importance for Delay Prediction',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
    return fig


def plot_weather_impact(df):
    """Bar chart: Weather impact on delays"""
    
    weather_delay = df.groupby('Weather_Impact')['Delayed'].mean() * 100
    weather_delay = weather_delay.sort_values(ascending=False)
    
    fig = px.bar(
        x=weather_delay.index,
        y=weather_delay.values,
        title='Delay Rate by Weather Condition',
        labels={'x': 'Weather Condition', 'y': 'Delay Rate (%)'},
        color=weather_delay.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400)
    return fig


def plot_distance_vs_delay(df):
    """Box plot: Distance categories vs delay"""
    
    # Create distance bins
    df_copy = df.copy()
    df_copy['Distance_Category'] = pd.cut(
        df_copy['Distance_KM'],
        bins=[0, 500, 1500, 3000, 5000],
        labels=['Short (<500km)', 'Medium (500-1500km)', 'Long (1500-3000km)', 'Very Long (>3000km)']
    )
    
    fig = px.box(
        df_copy,
        x='Distance_Category',
        y='Delay_Days',
        title='Delay Days by Distance Category',
        labels={'Distance_Category': 'Distance Category', 'Delay_Days': 'Delay (days)'},
        color='Distance_Category'
    )
    
    fig.update_layout(height=450, showlegend=False)
    return fig


def plot_cost_breakdown(df):
    """Stacked bar: Average cost breakdown"""
    
    cost_cols = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 
                 'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead']
    
    avg_costs = df[cost_cols].mean()
    
    fig = go.Figure(data=[
        go.Bar(name=col.replace('_', ' '), x=['Average Cost'], y=[avg_costs[col]])
        for col in cost_cols
    ])
    
    fig.update_layout(
        title='Average Cost Breakdown per Order',
        barmode='stack',
        yaxis_title='Cost (₹)',
        height=450
    )
    
    return fig


def create_feedback_wordcloud(df):
    """Generate word cloud from customer feedback"""
    
    # Check if feedback text exists
    if 'Feedback_Text' not in df.columns:
        return None
    
    feedback_text = ' '.join(df['Feedback_Text'].dropna().astype(str))
    
    if len(feedback_text) < 10:
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate(feedback_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Customer Feedback Word Cloud', fontsize=16, pad=20)
    
    return fig


def plot_timeline_trend(df):
    """Line chart: Delivery performance over time"""
    
    if 'Order_Date' not in df.columns:
        return None
    
    df_copy = df.copy()
    df_copy['Order_Date'] = pd.to_datetime(df_copy['Order_Date'])
    
    daily_metrics = df_copy.groupby(df_copy['Order_Date'].dt.date).agg({
        'On_Time': 'mean',
        'Customer_Rating': 'mean'
    }).reset_index()
    
    daily_metrics['On_Time'] = daily_metrics['On_Time'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_metrics['Order_Date'],
        y=daily_metrics['On_Time'],
        mode='lines+markers',
        name='On-Time %',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_metrics['Order_Date'],
        y=daily_metrics['Customer_Rating'] * 20,
        mode='lines+markers',
        name='Rating (scaled)',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='Delivery Performance Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Percentage / Scaled Rating',
        height=450
    )
    
    return fig