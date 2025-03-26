# import pickle
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# #from sklearn.pipeline import FeatureUnion  # Optional: For combining features
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# # Helper function for text preprocessing
# def preprocess_text(text):
#     return text.lower().strip()

# # Load dyslexia dataset (ensure the CSV file is located at 'dataset/dyslexia_text_samples.csv')
# df = pd.read_csv("dataset/dyslexia_text_samples.csv")  # Columns: 'text', 'label'

# # Preprocess text for consistency
# df["text"] = df["text"].apply(preprocess_text)

# X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# # Use character-level n-grams to capture subtle errors
# vectorizer = TfidfVectorizer(
#     max_features=1000,
#     analyzer='char',
#     ngram_range=(2, 4),
#     lowercase=True
# )

# # Optional: Combine with word-level features
# # word_vectorizer = TfidfVectorizer(max_features=1000, analyzer='word', ngram_range=(1, 2), lowercase=True)
# # combined_vectorizer = FeatureUnion([
# #     ("char", vectorizer),
# #     ("word", word_vectorizer)
# # ])
# # X_train_tfidf = combined_vectorizer.fit_transform(X_train)
# # X_test_tfidf = combined_vectorizer.transform(X_test)
# # Use combined_vectorizer in place of vectorizer below if desired.

# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Train a simple logistic regression classifier
# model = LogisticRegression()
# model.fit(X_train_tfidf, y_train)

# # Save model and vectorizer
# with open("models/dyslexia_text_model.pkl", "wb") as f:
#     pickle.dump((model, vectorizer), f)

# print("Text model trained and saved.")


import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

# Helper function for text preprocessing
def preprocess_text(text):
    return text.lower().strip()

# Load dyslexia dataset
df = pd.read_csv("dataset/dyslexia_text_samples.csv")  # Columns: 'text', 'label'

# Preprocess text for consistency
df["text"] = df["text"].apply(preprocess_text)

# Handle dataset imbalance if needed
class_weights = compute_class_weight("balanced", classes=np.unique(df["label"]), y=df["label"])
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Use both character and word n-grams for better generalization
char_vectorizer = TfidfVectorizer(max_features=1000, analyzer='char', ngram_range=(2, 4))
word_vectorizer = TfidfVectorizer(max_features=1000, analyzer='word', ngram_range=(1, 2))

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)
X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

# Combine both feature sets
from scipy.sparse import hstack
X_train_tfidf = hstack([X_train_char, X_train_word])
X_test_tfidf = hstack([X_test_char, X_test_word])

# Train a logistic regression model with SGD for multiple epochs
model = SGDClassifier(loss='log_loss', class_weight=class_weight_dict, max_iter=1, warm_start=True)


epochs = 10
for epoch in range(epochs):
    model.fit(X_train_tfidf, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_tfidf))
    test_acc = accuracy_score(y_test, model.predict(X_test_tfidf))
    print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Save model and vectorizers
with open("models/dyslexia_text_model.pkl", "wb") as f:
    pickle.dump((model, char_vectorizer, word_vectorizer), f)

print("✅ NLP model trained and saved successfully!")
