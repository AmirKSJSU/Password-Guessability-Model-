import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Embedding
import math

# Load the dataset
data = pd.read_csv("password_dataset.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['password'], data['strength'], test_size=0.2, random_state=42)

# Vectorize the passwords using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_vec, y_train)

# Evaluate the classifier
y_pred = rf_classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Classifier Accuracy:", accuracy)

# Define a neural network model
model = Sequential([
    Embedding(input_dim=len(vectorizer.vocabulary_), output_dim=50, input_length=X_train_vec.shape[1]),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the neural network
model.fit(X_train_vec, (y_train == 'Strong').astype(int), epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the neural network
loss, accuracy = model.evaluate(X_test_vec, (y_test == 'Strong').astype(int))
print("Neural Network Accuracy:", accuracy)

def password_entropy(password):
    """Calculate the entropy of a password."""
    password_length = len(password)
    character_set = len(set(password))
    entropy = password_length * math.log(character_set, 2) if character_set > 0 else 0
    return entropy

def feedback_for_weak_password(password):
    """Provide feedback for weak passwords."""
    entropy = password_entropy(password)
    if entropy < 30:
        return "Consider adding more characters or using a mix of uppercase, lowercase, digits, and symbols."
    elif entropy < 40:
        return "Consider using a mix of uppercase, lowercase, digits, and symbols for better security."
    elif entropy < 50:
        return "Your password is okay, but could be stronger with more complexity."
    else:
        return "Your password is strong!"

def predict_password_strength(password):
    """Predict the strength of a password."""
    password_vec = vectorizer.transform([password])
    rf_prediction = rf_classifier.predict(password_vec)[0]
    nn_prediction = model.predict(password_vec)[0][0]
    strength = "Strong" if nn_prediction >= 0.5 else "Weak"
    return strength, rf_prediction, nn_prediction

# Example usage
password = input("Enter a password to check its strength: ")
strength, rf_prediction, nn_prediction = predict_password_strength(password)
feedback = feedback_for_weak_password(password) if strength == "Weak" else ""
print("Password Strength (Random Forest):", strength)
print("Password Strength (Neural Network):", "Strong" if nn_prediction >= 0.5 else "Weak")
print("Feedback:", feedback)
