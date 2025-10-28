"""
Predictive Delivery Optimizer - Streamlit App
NexGen Logistics Pvt. Ltd.

Main application file for the delivery optimization dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_all_data, prepare_ml_data
from model_training import (
    train_model, predict_delay, get_recommendation, 
    analyze_feature_importance, get_carrier_performance, optimize_vehicle_selection
)
from visuals import (
    plot_kpi_cards, plot_delay_by_priority, plot_cost_vs_delay,
    plot_rating_distribution, plot_delivery_status_pie, plot_carrier_performance,
    plot_correlation_heatmap, plot_feature_importance, plot_weather_impact,
    plot_distance_vs_delay, plot_cost_breakdown, create_feedback_wordcloud,
    plot_timeline_trend
)

# Page configuration
st.set_page_config(
    page_title="NexGen Delivery Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.model_trained = False


def load_data():
    """Load and preprocess data"""
    with st.spinner("Loading and processing data..."):
        df, kpis = preprocess_all_data()
        
        if df is not None:
            st.session_state.df = df
            st.session_state.kpis = kpis
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
        else:
            st.error("❌ Failed to load data. Please check file paths.")


def train_ml_model():
    """Train machine learning model"""
    with st.spinner("Training ML model..."):
        X, y, feature_cols = prepare_ml_data(st.session_state.df)
        
        if len(X) > 0:
            model, scaler, feature_importance, metrics = train_model(X, y)
            
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_importance = feature_importance
            st.session_state.metrics = metrics
            st.session_state.feature_cols = feature_cols
            st.session_state.model_trained = True
            
            st.success(f"✅ Model trained! Accuracy: {metrics['accuracy']:.2%}")
        else:
            st.error("❌ Insufficient data for model training")


# Sidebar
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("### 🚚 Delivery Optimizer")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigate to:",
        ["📊 Overview Dashboard", "🔮 Predict Delay", "📈 Insights & Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Data loading
    if not st.session_state.data_loaded:
        if st.button("🔄 Load Data", use_container_width=True):
            load_data()
    else:
        st.success("✅ Data Loaded")
        
        if not st.session_state.model_trained:
            if st.button("🧠 Train ML Model", use_container_width=True):
                train_ml_model()
        else:
            st.success("✅ Model Ready")
    
    st.markdown("---")
    st.markdown("**About**")
    st.info("This dashboard helps optimize delivery operations using data analytics and machine learning.")


# Main content
if page == "📊 Overview Dashboard":
    st.markdown('<div class="main-header">📊 Delivery Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time insights into NexGen Logistics operations</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data using the sidebar button")
    else:
        df = st.session_state.df
        kpis = st.session_state.kpis
        
        # KPI Cards
        st.markdown("### 📈 Key Performance Indicators")
        plot_kpi_cards(kpis)
        
        st.markdown("---")
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Orders", f"{kpis['total_orders']:,}")
        with col2:
            st.metric("Severe Delays", f"{kpis['severe_delays']:,}")
        with col3:
            st.metric("Avg Distance", f"{kpis['avg_distance']:.0f} km")
        
        st.markdown("---")
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_delay_by_priority(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_delivery_status_pie(df), use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_rating_distribution(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_carrier_performance(df), use_container_width=True)
        
        # Charts row 3
        st.plotly_chart(plot_cost_vs_delay(df), use_container_width=True)
        
        # Carrier performance table
        st.markdown("### 🚛 Carrier Performance Analysis")
        carrier_stats = get_carrier_performance(df)
        st.dataframe(
            carrier_stats.style.background_gradient(cmap='RdYlGn_r', subset=['Delay_Rate'])
                              .background_gradient(cmap='RdYlGn', subset=['Avg_Rating']),
            use_container_width=True
        )


elif page == "🔮 Predict Delay":
    st.markdown('<div class="main-header">🔮 Delay Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict delivery delays for new orders</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
    elif not st.session_state.model_trained:
        st.warning("⚠️ Please train the ML model first")
    else:
        st.markdown("### 📝 Enter Order Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            distance = st.number_input(
                "Distance (km)",
                min_value=0.0,
                max_value=5000.0,
                value=500.0,
                step=50.0
            )
            
            priority = st.selectbox(
                "Priority Level",
                ["Express", "Standard", "Economy"]
            )
            
            traffic_delay = st.slider(
                "Expected Traffic Delay (minutes)",
                min_value=0,
                max_value=120,
                value=30
            )
        
        with col2:
            weather_impact = st.selectbox(
                "Weather Condition",
                ["None", "Light_Rain", "Heavy_Rain", "Fog"]
            )
            
            toll_charges = st.number_input(
                "Toll Charges (₹)",
                min_value=0.0,
                max_value=2000.0,
                value=300.0,
                step=50.0
            )
            
            fuel_consumption = st.number_input(
                "Expected Fuel Consumption (L)",
                min_value=0.0,
                max_value=600.0,
                value=50.0,
                step=5.0
            )
        
        with col3:
            order_value = st.number_input(
                "Order Value (₹)",
                min_value=0.0,
                max_value=50000.0,
                value=1000.0,
                step=100.0
            )
            
            st.markdown("### 🚛 Vehicle Suggestion")
            suggested_vehicle, reason = optimize_vehicle_selection(distance, priority)
            st.info(f"**Recommended:** {suggested_vehicle}\n\n{reason}")
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Delay Risk", use_container_width=True, type="primary"):
            # Prepare input
            priority_encoded = {'Express': 2, 'Standard': 1, 'Economy': 0}[priority]
            has_weather = 1 if weather_impact != 'None' else 0
            
            input_features = [
                distance,
                priority_encoded,
                traffic_delay,
                has_weather,
                toll_charges,
                fuel_consumption,
                order_value
            ]
            
            # Make prediction
            prediction, probability = predict_delay(
                st.session_state.model,
                st.session_state.scaler,
                input_features
            )
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if probability > 0.7:
                    st.error(f"### ⚠️ HIGH RISK")
                elif probability > 0.4:
                    st.warning(f"### ⚡ MODERATE RISK")
                else:
                    st.success(f"### ✅ LOW RISK")
                
                st.metric(
                    "Delay Probability",
                    f"{probability:.1%}",
                    delta=f"{probability - 0.5:.1%} vs baseline"
                )
                
                # Progress bar
                st.progress(probability)
            
            with col2:
                st.markdown("### 💡 Recommendations")
                recommendations = get_recommendation(probability, distance, priority)
                for rec in recommendations:
                    st.markdown(rec)
            
            # Feature contribution
            st.markdown("---")
            st.markdown("### 📊 Input Summary")
            
            input_df = pd.DataFrame({
                'Feature': st.session_state.feature_cols,
                'Value': input_features
            })
            
            st.dataframe(input_df, use_container_width=True)


elif page == "📈 Insights & Analytics":
    st.markdown('<div class="main-header">📈 Advanced Analytics & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep dive into delivery performance patterns</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
    else:
        df = st.session_state.df
        
        # Tabs for different insights
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Feature Analysis", "🌤️ Weather & Distance", "💰 Cost Analysis", "💬 Customer Feedback"])
        
        with tab1:
            if st.session_state.model_trained:
                st.markdown("### 🎯 Feature Importance")
                st.plotly_chart(
                    plot_feature_importance(st.session_state.feature_importance),
                    use_container_width=True
                )
                
                # Insights
                st.markdown("### 💡 Key Insights")
                insights = analyze_feature_importance(st.session_state.feature_importance)
                for insight in insights:
                    st.info(insight)
                
                # Model metrics
                st.markdown("### 📊 Model Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{st.session_state.metrics['accuracy']:.2%}")
                with col2:
                    if st.session_state.metrics['roc_auc']:
                        st.metric("ROC-AUC", f"{st.session_state.metrics['roc_auc']:.2%}")
                with col3:
                    st.metric("Model Type", "Random Forest")
            else:
                st.warning("⚠️ Train the model to see feature importance")
            
            # Correlation heatmap
            st.markdown("### 🔗 Feature Correlations")
            fig = plot_correlation_heatmap(df)
            st.pyplot(fig)
        
        with tab2:
            st.markdown("### 🌤️ Weather Impact Analysis")
            st.plotly_chart(plot_weather_impact(df), use_container_width=True)
            
            st.markdown("### 📏 Distance vs Delay Analysis")
            st.plotly_chart(plot_distance_vs_delay(df), use_container_width=True)
            
            # Timeline trend
            timeline_fig = plot_timeline_trend(df)
            if timeline_fig:
                st.markdown("### 📅 Performance Trend Over Time")
                st.plotly_chart(timeline_fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 💰 Cost Breakdown Analysis")
            st.plotly_chart(plot_cost_breakdown(df), use_container_width=True)
            
            # Cost statistics
            st.markdown("### 📊 Cost Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Total Cost", f"₹{df['Total_Cost'].mean():.2f}")
            with col2:
                st.metric("Avg Cost per KM", f"₹{df['Cost_Per_KM'].mean():.2f}")
            with col3:
                st.metric("Total Fuel Cost", f"₹{df['Fuel_Cost'].sum():.2f}")
            
            # Cost by priority
            st.markdown("### 💼 Cost Analysis by Priority")
            cost_priority = df.groupby('Priority')['Total_Cost'].agg(['mean', 'median', 'std']).round(2)
            st.dataframe(cost_priority, use_container_width=True)
        
        with tab4:
            st.markdown("### 💬 Customer Feedback Analysis")
            
            # Feedback statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_rating = df['Rating'].mean() if 'Rating' in df.columns else df['Customer_Rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.2f}")
            
            with col2:
                if 'Would_Recommend' in df.columns:
                    recommend_pct = (df['Would_Recommend'] == 'Yes').sum() / len(df) * 100
                    st.metric("Would Recommend", f"{recommend_pct:.1f}%")
            
            with col3:
                if 'Issue_Category' in df.columns:
                    top_issue = df['Issue_Category'].mode()[0]
                    st.metric("Top Issue", top_issue)
            
            # Word cloud
            wordcloud_fig = create_feedback_wordcloud(df)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.info("💡 Feedback word cloud not available")
            
            # Issue breakdown
            if 'Issue_Category' in df.columns:
                st.markdown("### 🔍 Issue Category Breakdown")
                issue_counts = df['Issue_Category'].value_counts()
                st.bar_chart(issue_counts)


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>NexGen Logistics Pvt. Ltd. - Predictive Delivery Optimizer</p>
        <p>Powered by Machine Learning & Data Analytics</p>
    </div>
""", unsafe_allow_html=True)