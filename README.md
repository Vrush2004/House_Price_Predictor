# House Price Prediction Project

## Overview
This project uses Linear Regression, a supervised learning model, to predict house prices based on:
- Square footage
- Number of bathrooms
- Number of bedrooms (BHK)
- Location

Developed using a Flask server.

## Workflow

### 1. Data Preprocessing
- Clean data by removing unnecessary columns and handling missing values.
- Feature engineering to extract useful information (e.g., converting non-numeric total square footage values).

### 2. Outlier Removal
- Remove outliers based on statistical analysis of price per square foot and BHK-specific data.

### 3. Model Training
- Split the dataset into training and test sets.
- Train a Linear Regression model.
- Evaluate model performance using cross-validation.

### 4. Model Selection
- Use GridSearchCV to find the best model among:
  - Linear Regression
  - Lasso Regression
  - Decision Tree Regression

### 5. Deployment
- Serialize the trained model using pickle.
- Develop a prediction function to estimate house prices based on user inputs.
- Save the model and column data for deployment with Flask.

## Getting Started

### Prerequisites
- Python 3.x
- Flask
- Pandas
- Numpy
- Scikit-learn
- Matplotlib

### Running the Project
1. Start the Flask server:
    ```sh
    python app.py
    ```
2. Open your web browser and go to `http://127.0.0.1:5000/` to use the house price prediction application.
<hr>
<br>

![Output](https://github.com/Vrush2004/House_Price_Predictor/assets/131949619/3c19aad4-5a8f-4df2-b9b3-59ee15318015)

