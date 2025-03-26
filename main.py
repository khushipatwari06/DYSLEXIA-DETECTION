import sys
print(f"Python Path: {sys.executable}")  # Should point to dyslexia_env

from scripts.preprocess import load_data
from scripts.train import DyslexiaGNN, train_model
from scripts.test import test_model
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Helper function to preprocess text (ensuring consistency)
def preprocess_text(text):
    return text.lower().strip()

# ------------------------------
# Image-Based Model Training
# ------------------------------

# Load dataset for image classification
train_loader, test_loader = load_data()

# Initialize GNN model for handwriting analysis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gnn_model = DyslexiaGNN(in_channels=1, hidden_channels=128, out_channels=64).to(device)

# Use BCEWithLogitsLoss for stable training
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(gnn_model.parameters(), lr=0.0005, weight_decay=1e-4)

# Train GNN model
print("Training Graph Neural Network for handwriting analysis...")
train_model(gnn_model, train_loader, criterion, optimizer, epochs=30)

# Test GNN model
print("Testing GNN model...")
test_model(gnn_model, test_loader)

# Save GNN model
torch.save(gnn_model.state_dict(), "models/dyslexia_gnn.pth")
print("✅ GNN model saved successfully!")

# ------------------------------
# Text-Based Model Training
# ------------------------------

print("\nTraining NLP Model for text-based dyslexia detection...")

# Load dataset for text-based detection (ensure the CSV is at 'dataset/dyslexia_text_samples.csv')
df = pd.read_csv("dataset/dyslexia_text_samples.csv")  # Columns: 'text', 'label'

# Preprocess text in the dataset for consistency
df["text"] = df["text"].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Use character-level n-grams to capture dyslexic error patterns
vectorizer = TfidfVectorizer(
    max_features=1000,
    analyzer='char',
    ngram_range=(2, 4),
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a simple logistic regression classifier
text_model = LogisticRegression()
text_model.fit(X_train_tfidf, y_train)

# Save NLP model and vectorizer
with open("models/dyslexia_text_model.pkl", "wb") as f:
    pickle.dump((text_model, vectorizer), f)

print("✅ NLP model trained and saved successfully!")
print("\nAll models (GNN and NLP) are now ready for use in dyslexia detection.")
