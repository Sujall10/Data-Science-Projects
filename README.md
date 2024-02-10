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

## Author

This project is maintained by Sujal Rajput.

## Issues

If you encounter any errors or have suggestions for improvements, feel free to contact me at [sujal0710rajput@gmail.com](mailto:sujal0710rajput@gmail.com).


### Stay tuned for the daily updates!
