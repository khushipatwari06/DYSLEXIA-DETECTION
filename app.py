from flask import Flask, request, jsonify, render_template
import torch
import pickle
from torch_geometric.data import Data
from scripts.train import DyslexiaGNN
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# Load trained image model
model_path = "models/dyslexia_gnn.pth"
input_dim, hidden_dim, output_dim = 1, 128, 64
image_model = DyslexiaGNN(input_dim, hidden_dim, output_dim)
image_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
image_model.eval()

# Load trained text model and vectorizer
with open("models/dyslexia_text_model.pkl", "rb") as f:
    text_model, vectorizer = pickle.load(f)
    # text_model, char_vectorizer, word_vectorizer = pickle.load(f)

# Helper function for text preprocessing (must match training)
def preprocess_text(text):
    return text.lower().strip()

# Image preprocessing function
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# Convert image to graph representation for GNN
def image_to_graph(image_tensor):
    image = image_tensor.squeeze().numpy()
    height, width = image.shape
    edges = [(i * width + j, ni * width + nj) 
             for i in range(height) for j in range(width) 
             for di in [-1, 0, 1] for dj in [-1, 0, 1] 
             if (di, dj) != (0, 0) and 0 <= (ni := i + di) < height and 0 <= (nj := j + dj) < width]
    return Data(x=torch.tensor(image.flatten(), dtype=torch.float).view(-1, 1),
                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image = Image.open(file).convert("RGB")
    graph_data = image_to_graph(transform_image(image).squeeze(0))
    
    with torch.no_grad():
        prob = torch.sigmoid(image_model(graph_data)).item()
    
    return jsonify({"prediction": "Non-dyslexic handwriting" if prob > 0.5 else "Dyslexic handwriting"})
@app.route("/predict_text", methods=["POST"])
def predict_text():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Preprocess the incoming text to match training conditions
    text = preprocess_text(text)
    
    # Convert text to TF-IDF features using the loaded vectorizer
    text_features = vectorizer.transform([text])

    # Get prediction probability for dyslexic (class 1)
    prob = text_model.predict_proba(text_features)[0][1]
    print("Predicted probability for dyslexic:", prob)  # Diagnostic logging
    
    # Adjust threshold if needed; default is 0.5
    threshold = 0.5
    prediction = "Dyslexic writing" if prob >= threshold else "Non-dyslexic writing"
    
    return jsonify({"prediction": prediction, "probability": prob})

if __name__ == "__main__":
    app.run(debug=True)

