import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Define Columns
text_column = "Name"  # Using 'Name' as a text feature
categorical_columns = ["Sex", "Embarked"]
numerical_columns = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
target_columns = ["Survived", "Pclass"]  # Multi-label target (assumed)

df.dropna(inplace=True)  # Drop missing values

# -----------------------------
# 2. Preprocess Text Data
# -----------------------------
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

tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df[text_column]).toarray()

# -----------------------------
# 3. Encode Categorical Data
# -----------------------------
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[categorical_columns])

# -----------------------------
# 4. Scale Numerical Data
# -----------------------------
scaler = StandardScaler()
X_num = scaler.fit_transform(df[numerical_columns])

# -----------------------------
# 5. Prepare Multi-Label Target
# -----------------------------
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df[target_columns].values)


# -----------------------------
# 6. Concatenate Features
# -----------------------------
X = np.hstack((X_text, X_cat, X_num))

# -----------------------------
# 7. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 8. Train Multi-Label Model (Random Forest)
# -----------------------------
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
multi_label_model = MultiOutputClassifier(base_model)  # Handles multi-label classification

multi_label_model.fit(X_train, y_train)

# -----------------------------
# 9. Evaluate Model
# -----------------------------
y_pred = multi_label_model.predict(X_test)

accuracy = np.mean([accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])])
print(f"Model Accuracy (Average across labels): {accuracy:.4f}")
