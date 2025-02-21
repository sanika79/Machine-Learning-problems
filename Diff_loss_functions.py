import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence, MeanSquaredError
from tensorflow.keras.utils import to_categorical

# Sample dataset
data = {
    'numerical_feature': [1.2, 3.4, 2.1, 5.6, 7.8, 4.3, 6.7, 8.9],
    'categorical_feature': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
    'text_feature': ['good service', 'bad experience', 'excellent product', 'average', 
                     'poor quality', 'best ever', 'worst ever', 'very nice'],
    'label': [0, 1, 2, 1, 0, 2, 1, 0]  # Multi-class labels
}
df = pd.DataFrame(data)

# Splitting dataset
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Preprocessing numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train[['numerical_feature']])
X_test_num = scaler.transform(test[['numerical_feature']])

# Preprocessing categorical features
encoder = OneHotEncoder(sparse=False)
X_train_cat = encoder.fit_transform(train[['categorical_feature']])
X_test_cat = encoder.transform(test[['categorical_feature']])

# Preprocessing textual features using TF-IDF
vectorizer = TfidfVectorizer(max_features=10)
X_train_text = vectorizer.fit_transform(train['text_feature']).toarray()
X_test_text = vectorizer.transform(test['text_feature']).toarray()

# Combine all features
X_train = np.hstack([X_train_num, X_train_cat, X_train_text])
X_test = np.hstack([X_test_num, X_test_cat, X_test_text])

# Encode labels for multi-class classification
y_train = to_categorical(train['label'])
y_test = to_categorical(test['label'])

# Build ML model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')  # Multi-class classification
])

# Experimenting with different loss functions
loss_functions = {
    'categorical_crossentropy': CategoricalCrossentropy(),
    'kl_divergence': KLDivergence(),
    'mean_squared_error': MeanSquaredError()
}

for loss_name, loss_fn in loss_functions.items():
    print(f"\nTraining with loss function: {loss_name}")
    
    model.compile(optimizer=Adam(learning_rate=0.01), loss=loss_fn, metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test), verbose=1)

    results = model.evaluate(X_test, y_test)
    print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
