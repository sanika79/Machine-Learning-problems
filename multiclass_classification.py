import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------
# 1. Load Dataset
# ---------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"  # Replace with actual dataset
df = pd.read_csv(url)

# Display sample
print("Dataset Sample:\n", df.head())

# Assume dataset has:
text_column = "Name"  # Replace with actual text column
categorical_columns = ["Pclass", "Sex", "Embarked"]  # Replace with actual categorical columns
numerical_columns = ["Age", "Fare"]  # Replace with actual numerical columns
target_column = "Survived"  # Replace with actual multi-class target column

# Drop missing values
df.dropna(inplace=True)

# ---------------------------------
# 2. Text Preprocessing
# ---------------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df[text_column] = df[text_column].astype(str).apply(clean_text)

# ---------------------------------
# 3. Feature Engineering
# ---------------------------------

# Encode Categorical Variables using OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_categorical = encoder.fit_transform(df[categorical_columns])

# Scale Numerical Data
scaler = StandardScaler()
X_numerical = scaler.fit_transform(df[numerical_columns])

# Convert Text to TF-IDF Features
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df[text_column]).toarray()

# Prepare Final Features
X = np.hstack((X_text, X_categorical, X_numerical))
y = LabelEncoder().fit_transform(df[target_column])  # Encode target variable

# ---------------------------------
# 4. Train Model
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------
# 5. Evaluate Model
# ---------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
