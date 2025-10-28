# NexGen Delivery Optimizer

### Logistics Innovation Challenge — NexGen Logistics Pvt. Ltd.

**Tech Stack:** Python | Streamlit | Pandas | Scikit-learn | Plotly | WordCloud  
**Machine Learning Model:** Gradient Boosting Classifier  
**Innovation Area:** Predictive Logistics & Smart Recommendations  


## Keywords
Predictive Analytics · Gradient Boosting · Streamlit Dashboard · Data Visualization · Recommendation Engine · Cost Optimization · Route & Fleet Insights · Logistic Delay Prediction · Customer Experience Analysis


## Project Overview
NexGen Logistics faces challenges such as delivery delays, rising costs, and lack of predictive intelligence.  
**NexGen Delivery Optimizer** is a data-driven analytics and ML platform that predicts delivery delays **before they occur**, provides **actionable insights**, and recommends **optimal operational decisions** in real-time.

The solution helps NexGen move from **reactive** to **predictive** operations, improving **efficiency, cost control, and customer satisfaction.**


## 🧩 Project Structure

nexgen_delivery_optimizer/
├── data/
│ ├── orders.csv
│ ├── delivery_performance.csv
│ ├── routes_distance.csv
│ ├── vehicle_fleet.csv
│ ├── cost_breakdown.csv
│ └── customer_feedback.csv
│
├── app.py
├── data_preprocessing.py
├── model_training.py
├── visuals.py
├── requirements.txt
└── README.md



##  Components & Functionality

### 1. **Data Preprocessing (`data_preprocessing.py`)**
- Loads and cleans all datasets (handles missing values, duplicates)
- Merges six CSVs into a unified logistics dataset  
- Engineers new features (e.g., delay flags, cost ratios)
- Computes KPIs such as:
  - On-Time Delivery Rate
  - Average Delay
  - Customer Ratings
  - Average Cost per Delivery



### 2. **Model Training (`model_training.py`)**
- Uses **Gradient Boosting Classifier** for predicting delivery delays  
  *(chosen over Random Forest for better performance on small, imbalanced datasets)*
- Includes:
  - Train/test split and feature scaling  
  - Model accuracy, ROC-AUC, and feature importance  
  - Confusion matrix and detailed classification report  
- Integrates a **Rule-Based Recommendation Engine** that:
  - Suggests delivery strategies based on delay probability
  - Optimizes vehicle selection and carrier choice
  - Advises on operational adjustments (route, buffer, express upgrade)



### 3. **Visual Analytics (`visuals.py`)**
- Dynamic and interactive visualizations:
  - KPIs and metric cards  
  - Delay analysis by priority  
  - Cost vs Delay scatter plots  
  - Carrier performance heatmaps  
  - Customer ratings and feedback word clouds  
  - Weather and distance impact plots  
  - Feature importance charts  



### 4. **Streamlit App (`app.py`)**
#### **Main Sections:**
1. **📊 Overview Dashboard**
   - Real-time KPIs and visual summaries  
   - Priority-based delay trends  
   - Carrier comparison and cost distribution  

2. **🔮 Predict Delay**
   - User form for new order parameters  
   - Real-time prediction with probability output  
   - Smart recommendations and risk insights  
   - Vehicle optimization suggestions  

3. **📈 Insights & Analytics**
   - Feature analysis and model insights  
   - Weather and route impact trends  
   - Cost breakdown by priority and type  
   - Customer feedback visualization  

---

##  Innovation Highlights

###  Predictive Intelligence
- Uses ML to proactively forecast delivery risks instead of reacting post-failure.

###  Recommendation Engine
- Rule-based AI logic that generates **real-time operational suggestions** based on predicted risk, distance, and priority.

###  Cost Intelligence
- Merges cost breakdown and performance data to identify **optimization opportunities** (e.g., high fuel cost carriers, low efficiency routes).

###  Fleet & Route Optimization
- Suggests ideal vehicle types based on delivery type, distance, and priority.

###  Sustainability Support
- Indirectly reduces emissions and fuel use via smarter routing and fleet decisions.

---

##  Model Performance

| Metric | Value |
|--------|--------|
| Accuracy | ~47% *(small dataset — proof-of-concept)* |
| ROC-AUC | Moderate |
| Insights | Distance, Weather Impact, and Priority are top predictive factors |

**Note:** The dataset provided for the challenge was limited (≈200 rows).  
In real-world enterprise deployment with millions of records, the model would achieve significantly higher predictive accuracy due to richer feature diversity.



##  How to Run

### 1️ Clone Repository
git clone https://github.com/yourusername/nexgen_delivery_optimizer.git
cd nexgen_delivery_optimizer

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run app.py

| Goal                        | Achieved Through                    |
| --------------------------- | ----------------------------------- |
| Reduce operational delays   | ML-based delay prediction           |
| Improve cost efficiency     | Cost breakdown analytics            |
| Enhance customer experience | Feedback insights & recommendations |
| Build predictive culture    | Data-driven dashboards              |
| Sustainability              | Route & fleet optimization          |

📄 Acknowledgement

Developed as part of the Logistics Innovation Challenge by NexGen Logistics Pvt. Ltd.
Author: Anushka Das


📘 License

This project is developed for academic and demonstration purposes under the Logistics Innovation Challenge guidelines.
