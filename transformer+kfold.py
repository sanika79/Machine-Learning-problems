

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier

# Example data (replace with your dataset)
data = {
    'numerical': [1.2, 2.3, 3.1, 4.5, 5.6],
    'categorical': ['cat', 'dog', 'cat', 'dog', 'rabbit'],
    'text': ['dog barks', 'cat meows', 'dog runs', 'cat sleeps', 'rabbit hops'],
    'target': ['class1', 'class2', 'class1', 'class3', 'class2']
}

df = pd.DataFrame(data)

# Preprocessing: Numerical, Categorical, Text features
X = df.drop(columns=['target'])
y = df['target']

# Define categorical columns and numerical columns
categorical_cols = ['categorical']
numerical_cols = ['numerical']
text_cols = ['text']

# Define transformers for each type of data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_transformer = Pipeline(steps=[
    ('vectorizer', TfidfVectorizer())
])

# Combine all transformers into one preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('text', text_transformer, text_cols)
    ]
)

# Define the model: RandomForestClassifier for multi-class classification
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# KFold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)

# Output the average accuracy
print(f'Average Accuracy: {np.mean(accuracies):.4f}')
