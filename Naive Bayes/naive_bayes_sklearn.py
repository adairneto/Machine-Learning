import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
dataset = pd.read_csv("Finance.csv")
x = dataset.iloc[:, 0].values  # Assuming first column contains text
y = dataset.iloc[:, 1].values  # Assuming second column contains labels

# Split the dataset into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Text vectorization
vectorizer = CountVectorizer(stop_words='english')
x_train_transformed = vectorizer.fit_transform(x_train)
x_test_transformed = vectorizer.transform(x_test)

# Train the Naive Bayes model
classifier = MultinomialNB()
classifier.fit(x_train_transformed, y_train)

# Predict on test data
y_pred = classifier.predict(x_test_transformed)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
