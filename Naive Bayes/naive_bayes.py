import numpy as np
import pandas as pd
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Load and split dataset
dataset = pd.read_csv("Finance.csv")
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
np.random.seed(0)
indices = np.random.permutation(len(x))
train_size = int(0.75 * len(x))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
x_train, x_test = x[train_indices], x[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    return tokens

# Build vocabulary and word counts
def build_vocab_and_counts(x_train, y_train):
    vocab = set()
    word_counts = defaultdict(Counter)
    class_counts = Counter(y_train)
    for text, label in zip(x_train, y_train):
        tokens = preprocess_text(text)
        vocab.update(tokens)
        word_counts[label].update(tokens)
    return vocab, word_counts, class_counts

vocab, word_counts, class_counts = build_vocab_and_counts(x_train, y_train)

# Calculate probabilities
def calculate_probabilities(vocab, word_counts, class_counts):
    class_prob = {label: count / sum(class_counts.values()) for label, count in class_counts.items()}
    word_prob = {}
    for label, counter in word_counts.items():
        total_words = sum(counter.values())
        word_prob[label] = {word: (counter[word] + 1) / (total_words + len(vocab)) for word in vocab}
    return class_prob, word_prob

class_prob, word_prob = calculate_probabilities(vocab, word_counts, class_counts)

# Predict function
def predict(text, class_prob, word_prob, vocab):
    tokens = preprocess_text(text)
    scores = {}
    for label in class_prob:
        scores[label] = np.log(class_prob[label])
        for token in tokens:
            if token in vocab:
                scores[label] += np.log(word_prob[label].get(token, 1 / len(vocab)))
    return max(scores, key=scores.get)

# Predictions and evaluation
y_pred = [predict(text, class_prob, word_prob, vocab) for text in x_test]
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
confusion_matrix = pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(y_pred, name="Predicted"))
print("\nConfusion Matrix:\n", confusion_matrix)
confusion_matrix.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Confusion Matrix")
plt.xlabel("Actual")
plt.ylabel("Frequency")
plt.show()
