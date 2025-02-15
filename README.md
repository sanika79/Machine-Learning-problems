Problem 1
Problem Statement
You are given a dataset with different types of data (numerical, categorical, and text). Your task is to preprocess the features appropriately and train a machine learning model to predict a target variable.

Dataset Description
The dataset contains the following columns:

age (Numerical) - Age of the person.
income (Numerical) - Annual income in USD.
education (Categorical) - Highest level of education (e.g., High School, Bachelor’s, Master’s, PhD).
city (Categorical) - City of residence.
job_title (Text) - Job description/title.
purchases_last_6m (Numerical) - Number of purchases made in the last 6 months.
target (Binary) - Whether the person is likely to buy a product (1 for Yes, 0 for No).

Expectations
Preprocess different types of data appropriately
Handle missing values, normalize numerical data, and encode categorical data
Extract features from text data (e.g., TF-IDF, embeddings)
Train a model (Logistic Regression, Decision Trees, or any other model)
Evaluate performance using classification metrics


Problem 2

Problem Statement
We will build a binary classification model to predict whether a customer will buy a product (target: 1 or 0) based on their age, income, city, education level, and job description.

Pipeline Overview
Load Data
Preprocessing
Handle missing values
Encode categorical features
Extract features from text
Scale numerical features
Train-Test Split
Model Training
Evaluation
Deploying the Model for Prediction
