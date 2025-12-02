# Project Title:                US Accidents – Predictive Modelling and EDA
# Team Members: 
# Name                          #Email                              # Role Number
 Umesh BP                      umeshbp@iisc.ac.in                    13-19-02-19-52-25-1-26101    
 Roupyajay Bhattacharya        roupyajayb@iisc.ac.in                 13-19-02-19-52-25-1-26258   
 Ninad Sitaram Phadnis         ninadphadnis@iisc.ac.in               13-19-02-19-52-25-1-26399
 Madipally Bhagath Chandra     BhagathChan1@iisc.ac.in               13-19-02-19-52-25-1-26243

# Problem Statement:             DataScience US acciedents analysis

# Brief description of the dataset(s):
This is a countrywide car accident dataset that covers 49 states of the USA. The accident data were collected from February 2016 to March 2023, using multiple APIs that provide streaming traffic incident (or event) data. These APIs broadcast traffic data captured by various entities, including the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road networks. The dataset currently contains approximately 7.7 million accident records. 

# High-level approach and methods used
# EDA
1. Data Quality Assessment: Missing values, data types, data cleaning
2. Temporal Analysis: Trends by year, month, day, hour, season
3. Severity Distribution: Understanding the target variable (1-4 scale)
4. Geographic Patterns: State and city-level accident hotspots
5. Environmental Factors: Weather conditions, visibility, time of day
6. Infrastructure Impact: Road features (junctions, signals, roundabouts)
7. Bivariate Relationships: Correlations between features and severity
# Modelling
- Predict accident severity levels (1-4) based on environmental, temporal, and road infrastructure features
- Handle class imbalance using SMOTE-NC (for mixed categorical/numerical data)
- Optimize model performance through feature selection and hyperparameter tuning
- Provide interpretable results with probability estimates and feature importance analysis

# Summary of results
Logistic Regression:  Test Accuracy: 35%, Validation accuracy: 30%​
 Random Forest:  Test Accuracy: 40%, Validation accuracy: 42%​
XGBoost:  Test Accuracy: 50%, Validation accuracy: 50%​
Logistic Regression + Random Forest:  Test Accuracy: 60%, Validation accuracy: 65%​
Logistic Regression + Random Forest + XGBoost :  Test Accuracy: 89%, Validation accuracy: 84% (Final
model): Train F1 score: 0.9309​​

# Data Set: The data set is ~3 GB is size. Please refer the below link to download the data.
Kaggle link:
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

# App link : https://ds-us-accidents-analysis-7xhebyb48xunkr4cn8zqhe.streamlit.app/