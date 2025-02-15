import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # To save the trained model

# ------------------ STEP 1: Load Dataset ------------------
# Simulating a dataset
data = {
    'age': [25, 40, np.nan, 35, 50, 29, 42, 31, 28, np.nan],
    'income': [50000, 70000, 45000, 80000, 120000, 55000, 90000, np.nan, 67000, 58000],
    'education': ['Bachelors', 'Masters', 'PhD', 'High School', 'Bachelors', 'Masters', 'PhD', 'Bachelors', 'High School', 'Masters'],
    'city': ['New York', 'San Francisco', 'Chicago', 'Boston', 'New York', 'San Francisco', 'Chicago', 'Boston', 'New York', 'Chicago'],
    'job_title': ['Software Engineer', 'Data Scientist', 'Doctor', 'Teacher', 'Lawyer', 'Accountant', 'Engineer', 'Scientist', 'Nurse', 'Analyst'],
    'target': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# ------------------ STEP 2: Define Preprocessing Pipelines ------------------
numerical_features = ['age', 'income']
categorical_features = ['education', 'city']
text_feature = 'job_title'

# Pipeline for numerical data
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical data
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline for text data
text_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5))  # Extract top 5 important words
])

# Combine all preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features),
    ('txt', text_pipeline, text_feature)
])

# ------------------ STEP 3: Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ STEP 4: Model Training ------------------
# Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# ------------------ STEP 5: Model Evaluation ------------------
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ------------------ STEP 6: Deploying the Model for Prediction ------------------
# Save model
joblib.dump(model, "customer_prediction_model.pkl")

# Load model for prediction
loaded_model = joblib.load("customer_prediction_model.pkl")

# Sample new data point
new_data = pd.DataFrame({
    'age': [30],
    'income': [75000],
    'education': ['Masters'],
    'city': ['New York'],
    'job_title': ['Data Scientist']
})

# Make prediction
prediction = loaded_model.predict(new_data)
print("Predicted Class:", prediction[0])  # Output: 0 or 1
