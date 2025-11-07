# 21AIC401T_Customer-Churn-Case-Study        

## Project Overview
This project develops and compares machine learning models to predict customer churn in the telecom industry. The analysis includes data preprocessing, exploratory data analysis, model development (CHAID and Logistic Regression), model comparison, and deployment strategies.

## Course Information
- **Course**: Inferential Statistics and Predictive Analytics (21AIC401T)
- **Institution**: SRM University - Department of Computational Intelligence
- **Assignment Type**: Case Study-Based Modeling Project

## Dataset
- **Source**: Telco Customer Churn Dataset
- **Total Records**: [Your dataset size]
- **Features**: [Number of features]
- **Target Variable**: Churn (Yes/No)

## Repository Structure
```
├── Customer_Churn_Analysis.ipynb    # Main Jupyter notebook with complete analysis
├── cleaned_churn_dataset.csv        # Preprocessed dataset
├── churn_prediction_model.pkl       # Trained model (best performing)
├── scaler.pkl                       # Feature scaler
├── label_encoders.pkl               # Categorical encoders
├── feature_names.pkl                # Feature name list
├── deployment_script.py             # Model deployment script
├── model_update_guide.txt           # Model updating guidelines
├── model_results_summary.csv        # Performance metrics summary
├── feature_importance.csv           # Feature importance rankings
├── visualizations/                  # All generated charts and plots
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── roc_curves.png
│   ├── lift_chart.png
│   └── gains_chart.png
└── README.md                        # This file
```

## Key Findings

### 1. Data Insights
- **Churn Rate**: [X]% of customers churned
- **Key Churn Indicators**: [Top 3 features]
- **Customer Distribution**: [Brief summary]

### 2. Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| CHAID (Decision Tree) | [X.XX] | [X.XX] |
| Logistic Regression | [X.XX] | [X.XX] |

### 3. Top Predictive Features
1. [Feature 1]
2. [Feature 2]
3. [Feature 3]
4. [Feature 4]
5. [Feature 5]

## Technologies Used
- **Programming Language**: Python 3.x
- **Libraries**: 
  - pandas, numpy - Data manipulation
  - scikit-learn - Machine learning models
  - matplotlib, seaborn - Data visualization
  - joblib - Model serialization

## Installation and Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib imbalanced-learn
```

### Running the Analysis
1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Customer_Churn_Analysis.ipynb
```

3. Or run in Google Colab:
   - Upload the notebook to Google Drive
   - Open with Google Colab
   - Upload the dataset when prompted

## Model Deployment

### Loading the Model
```python
import joblib

# Load trained model
model = joblib.load('churn_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Make predictions
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)
```

### Using the Deployment Script
```bash
python deployment_script.py
```

## Model Updating Strategy

### Automated Retraining
- **Frequency**: Monthly or quarterly
- **Trigger**: Performance degradation (>5% drop in ROC-AUC)
- **Validation**: Hold-out test set and A/B testing

### Monitoring Metrics
- Accuracy
- ROC-AUC Score
- Precision and Recall
- Lift and Gains

## Business Recommendations

### Churn Prevention Strategies
1. **High-Risk Customers**: Target customers with churn probability >70%
2. **Retention Offers**: Customize offers based on key churn factors
3. **Early Warning System**: Monitor customers showing early churn signals
4. **Contract Incentives**: Encourage longer contract commitments

### Implementation Plan
- Deploy model as REST API or batch scoring system
- Integrate with CRM system for real-time predictions
- Create dashboard for monitoring churn risk
- Establish monthly review process for model performance

## Results and Visualizations

### Key Visualizations
1. **Churn Distribution**: Shows overall churn rate in dataset
2. **Feature Importance**: Identifies most influential factors
3. **ROC Curves**: Compares model discrimination ability
4. **Lift Chart**: Demonstrates model effectiveness
5. **Gains Chart**: Shows cumulative performance

## Limitations and Future Work

### Current Limitations
- Limited to available features in dataset
- Assumes static customer behavior patterns
- Does not account for external market factors

### Future Enhancements
- Incorporate time-series features for trend analysis
- Add customer lifetime value (CLV) calculations
- Implement ensemble methods (Random Forest, XGBoost)
- Develop customer segmentation for targeted strategies
- Create real-time prediction API

## Contributors
- [Your Name]
- [Registration Number]
- [Department of Computational Intelligence, SRM University]

## License
This project is submitted as part of academic coursework.

## Contact
For questions or feedback, please contact: [Your Email]

## Acknowledgments
- SRM University - Department of Computational Intelligence
- Course Instructor: [Instructor Name]
- Dataset Source: [Kaggle/UCI/Data.world]

---
**Submission Date**: 10.11.2025
**Course Code**: 21AIC401T
'''

with open('README.md', 'w') as f:
    f.write(readme_content)
