# 30-Day Data Science Challenge: Daily Projects

Welcome to the 30-day data science challenge! Each day, a new data science project will be uploaded to this repository. Each project will include loading data, splitting data, model building, model comparison, and data visualization.

## DAY1 : Delaney Solubility Prediction with Descriptors

This repository contains code for predicting solubility of chemical compounds using molecular descriptors. Below is an overview of the data analysis process:

### Data Loading

The initial step involves loading the Delaney solubility dataset, which includes chemical compounds alongside their solubility values and molecular descriptors. These descriptors furnish additional insights into the chemical characteristics of each compound. For this task, we'll employ data manipulation modules like Pandas for handling the dataset.

### Splitting Data

Subsequently, the dataset will be partitioned into training and testing sets, typically allocating 80% for training and the rest for testing. Modules like Scikit-learn's train_test_split function will facilitate this division.

### Model Building

Two predictive models will be constructed: Linear Regression and Random Forest.

#### Linear Regression

Utilizing Scikit-learn's LinearRegression module, we'll create a model to establish the relationship between solubility (dependent variable) and molecular descriptors (independent variables), deriving coefficients to optimize predictions.

#### Random Forest

Employing Scikit-learn's RandomForestRegressor, we'll employ an ensemble learning technique, constructing numerous decision trees during training to generate predictions by averaging individual tree outputs.

### Model Comparison

Post-training, the models will be assessed using evaluation metrics such as Mean Squared Error (MSE), and R-squared (R^2) to discern their predictive efficacy. This analysis will provide insights into which model performs better.

### Data Visualization for Prediction

To enhance comprehension of model predictions, visualization techniques will be applied. Scatter plots will be generated to contrast actual solubility values against predicted values, aiding in assessing model accuracy. Additionally, residual plots will help identify patterns or trends in prediction errors.

In conclusion, by following this comprehensive data analysis approach, we aim to develop reliable solubility prediction models using molecular descriptors, enabling informed decision-making in chemical research and development. Modules like Pandas, Scikit-learn, and Matplotlib will be instrumental throughout this project.


## DAY2 : Weather Prediction Data Science Project

Welcome to our Weather Prediction Data Science Project! In this project, we aim to predict weather conditions using machine learning techniques.

### Table of Contents
- [Data Loading](#data-loading)
- [Preparing Data for Machine Learning](#preparing-data-for-machine-learning)
- [Filling in Missing Values](#filling-in-missing-values)
- [Verifying Correct Data Types](#verifying-correct-data-types)
- [Analyzing Data](#analyzing-data)
- [Training Machine Learning Model](#training-machine-learning-model)
- [Evaluating Model](#evaluating-model)
- [Creating Prediction Function](#creating-prediction-function)
- [Adding in Rolling Means](#adding-in-rolling-means)

### Data Loading
We start by loading the weather dataset into our environment. Ensure you have the necessary libraries installed to manipulate data.

import pandas as pd

#### Load the weather dataset

weather_data = pd.read_csv("weather_data.csv")

### Preparing Data for Machine Learning

Next, we preprocess the data to make it suitable for machine learning algorithms. This involves handling missing values, encoding categorical variables, and scaling numerical features.

#### Filling in Missing Values

Missing values in the dataset can impact model performance. We handle missing data by imputing values using appropriate techniques such as mean, median, or interpolation.

#### Verifying Correct Data Types

Ensuring that data types are appropriate is crucial for accurate modeling. We verify and convert data types as needed to ensure consistency and accuracy.

### Analyzing Data
Understanding the data is essential for making informed decisions during model building. We explore various statistical summaries and visualizations to gain insights into the dataset.

### Training Machine Learning Model
We use Ridge regression, a linear regression algorithm regularized with L2 regularization, to train our weather prediction model. We set the regularization parameter alpha to 0.1.

### Evaluating Model
To assess the performance of our model, we evaluate it using appropriate evaluation metrics such as mean squared error, R-squared, or others depending on the nature of the problem.

#### Creating Prediction Function

Once the model is trained and evaluated, we create a function to make predictions on new data. This function takes input features and returns the predicted weather conditions.

#### Adding in Rolling Means

To improve the accuracy of our predictions, we incorporate rolling means of relevant features. This helps capture trends and seasonality in the data.

#### Stay tuned for updates as we continue to develop our weather prediction model!


## DAY3 : Olympics medal Prediction 

This project aims to predict the number of medals each country will win in a sports event using machine learning techniques.

### Table of Contents

- [Introduction](#introduction)
- [Steps Taken](#steps-taken)
  - [Loading the Data](#loading-the-data)
  - [Data Exploration](#data-exploration)
  - [Analyzing Data](#analyzing-data)
  - [Building the Model](#building-the-model)
  - [Measuring Error](#measuring-error)
  - [Checking Model Performance](#checking-model-performance)
- [Next Steps](#next-steps)

### Introduction

In this project, we use machine learning algorithms to predict the number of medals each country is likely to win.

### Steps Taken

#### Loading the Data

We sourced the data from [kaggle.com] which includes historical data on countries' performance. The dataset contains information such as country, year, sport, and medal count.

#### Data Exploration

We performed exploratory data analysis to understand the structure of the dataset and identify any patterns or trends. This involved summary statistics, data visualization, and identifying any missing or inconsistent data.

#### Analyzing Data

Based on the exploratory analysis, we identified key features that may influence a country's medal count, such as athlete, age, past performance. We also performed feature engineering to extract meaningful information from the raw data.

#### Building the Model

We experimented with various machine learning algorithms, including linear regression to build our prediction model.

#### Measuring Error

These metrics help quantify the difference between predicted and actual medal counts.

#### Checking Model Performance

We assessed the performance of our model using techniques such as train-test splits and k-fold cross-validation to ensure its generalizability to unseen data. We also compared our model's predictions against baseline models and evaluated its robustness to different scenarios.

### Next Steps

- Refine feature selection and engineering to improve model performance.
- Explore advanced machine learning techniques such as ensemble methods and neural networks.
- Incorporate external factors such as geopolitical events and athlete injuries into the prediction model.
- Deploy the model into production for real-time predictions during sports events.
- Continuously monitor and update the model to adapt to changing circumstances and improve accuracy.

Feel free to contribute or provide feedback on this project!



## Author

This project is maintained by Sujal Rajput.

## Issues

If you encounter any errors or have suggestions for improvements, feel free to contact me at [sujal0710rajput@gmail.com](mailto:sujal0710rajput@gmail.com).


### Stay tuned for the daily updates!
