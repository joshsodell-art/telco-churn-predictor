# Telco Customer Churn Predictor

![Feature Importance Plot](feature_importance.png)  
*(Top drivers of churn, like tenure and monthly charges — key for retention strategies!)*

### Project Overview
This end-to-end machine learning project predicts customer churn for telecom companies using customer data (e.g., tenure, services, charges). Churn is a big problem — it costs telecoms billions annually — so this model helps spot at-risk customers early for targeted interventions like discounts or upgrades.

Built from Week 1 notes as a hands-on accomplishment to strengthen ML fundamentals. Achieved ~79% accuracy with real-world insights on what keeps customers loyal.

### Value for Telecom
- **Reduces Churn Costs**: Flags high-risk customers (e.g., short-tenure, month-to-month contracts) before they leave, saving 5-25x the cost of acquiring new ones.
- **Boosts Retention**: Insights like "higher monthly charges increase churn risk" can inform personalized offers, improving customer satisfaction and lifetime value.
- **Drives Revenue**: Lower churn means steadier income; companies using similar ML models see 10-15% retention gains, per industry studies.

### Tech Stack
- Python (pandas, scikit-learn for modeling, matplotlib/seaborn for viz)
- Models: Logistic Regression (baseline) + Random Forest (tuned with GridSearchCV)

### Results & Insights
- **Logistic Regression**: 79.4% accuracy (solid baseline for binary prediction)
- **Tuned Random Forest**: 79.3% accuracy (optimized for F1 on imbalanced churn class)
- Key Findings: Tenure is the top predictor (longer = less churn); high monthly charges and month-to-month contracts are red flags. Full classification reports in the notebook.

### How to Run
1. Clone the repo: `git clone https://github.com/joshsodell-art/telco-churn-predictor.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook: `jupyter notebook churn_predictor.ipynb`
4. Run all cells — data loads automatically from URL!

### Code Structure
- Data loading & cleaning
- Preprocessing (encoding, scaling)
- Train/test split
- Modeling & tuning
- Evaluation (accuracy, F1, reports)
- Visualization (feature importance plot)

### Future Work
- Add advanced models like XGBoost for better accuracy
- Handle class imbalance with SMOTE
- Deploy as
