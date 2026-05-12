# 🏠 Boston House Price Prediction

## About
This project predicts the median value of owner-occupied homes in Boston 
using machine learning. The dataset contains 506 records and 13 socio-economic 
and geographic features.

## What We Did
- Loaded and explored the Boston Housing dataset
- Checked skewness of all features and applied Yeo-Johnson transformation
- No outlier removal — all values are real census data (Harrison & Rubinfeld, 1978)
- Split data into 80% train and 20% test
- Applied StandardScaler for feature scaling
- Checked multicollinearity using VIF — dropped nox and crim
- Compared 10 regression models — Linear Regression, Ridge, Lasso, 
  ElasticNet, Decision Tree, Random Forest, XGBoost, KNN, SVR, Voting Regressor
- XGBoost selected as best model
- Applied hyperparameter tuning
- Validated using 10-fold Cross Validation
- Deployed as interactive web app using Streamlit

## Model Performance
Train R² Score : 0.9832
Test R²  Score : 0.8882
Overfit Gap    : 0.095


## Tech Stack
Python, Scikit-learn, XGBoost, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Scipy


## Author
**Anushka Pal**  
📧 palanushka416@gmail.com  
🔗 [GitHub](https://github.com/AnushkaPal2003)
