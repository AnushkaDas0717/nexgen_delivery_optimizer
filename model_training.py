"""
ML Model Training Module
Handles model training, prediction, and feature importance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st


@st.cache_resource
def train_model(X, y):
    """Train Gradient Boosting model for delay prediction"""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gradient Boosting Model (better for small data)
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Cross-validation for robustness
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = None

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_accuracy': cv_mean,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    return model, scaler, feature_importance, metrics


def predict_delay(model, scaler, input_features):
    """Predict delay probability for new input"""
    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction, probability


def get_recommendation(probability, distance, priority):
    """Generate recommendation based on prediction"""
    recommendations = []

    if probability > 0.7:
        recommendations.append(" **HIGH DELAY RISK**")
        recommendations.append("• Consider upgrading to Express delivery")
        recommendations.append("• Assign to most reliable carrier")
        recommendations.append("• Add buffer time to promised delivery")
    elif probability > 0.4:
        recommendations.append(" **MODERATE DELAY RISK**")
        recommendations.append("• Monitor weather conditions closely")
        recommendations.append("• Use standard delivery protocol")
        recommendations.append("• Consider alternative routes if available")
    else:
        recommendations.append(" **LOW DELAY RISK**")
        recommendations.append("• Proceed with standard delivery")
        recommendations.append("• No special handling required")

    if distance > 2000:
        recommendations.append("• Long distance: Plan for rest stops")

    if priority == 'Express':
        recommendations.append("• Express priority: Ensure fastest route")

    return recommendations


def analyze_feature_importance(feature_importance):
    """Analyze and interpret feature importance"""
    insights = []
    top_feature = feature_importance.iloc[0]
    insights.append(f"**Top factor**: {top_feature['Feature']} ({top_feature['Importance']:.2%} importance)")

    if 'Distance_KM' in feature_importance['Feature'].values[:3]:
        insights.append("• Distance is a critical factor - optimize route planning")
    if 'Traffic_Delay_Minutes' in feature_importance['Feature'].values[:3]:
        insights.append("• Traffic delays significantly impact delivery - use real-time traffic data")
    if 'Has_Weather_Impact' in feature_importance['Feature'].values[:3]:
        insights.append("• Weather conditions matter - monitor forecasts closely")

    return insights


def get_carrier_performance(df):
    """Analyze carrier performance"""
    carrier_stats = df.groupby('Carrier').agg({
        'Delayed': 'mean',
        'Customer_Rating': 'mean',
        'Total_Cost': 'mean',
        'Order_ID': 'count'
    }).round(3)

    carrier_stats.columns = ['Delay_Rate', 'Avg_Rating', 'Avg_Cost', 'Total_Orders']
    carrier_stats = carrier_stats.sort_values('Delay_Rate')
    return carrier_stats


def optimize_vehicle_selection(distance, priority):
    """Suggest optimal vehicle type based on distance and priority"""
    if priority == 'Express' and distance < 500:
        return "Express_Bike", "Fast delivery for short distances"
    elif distance < 1000:
        return "Small_Van", "Cost-effective for medium distances"
    elif distance < 3000:
        return "Medium_Truck", "Balanced capacity and efficiency"
    else:
        return "Large_Truck", "High capacity for long distances"
