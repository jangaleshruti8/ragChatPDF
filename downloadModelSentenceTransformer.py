from sentence_transformers import SentenceTransformer
import os
import rag_config
# here define a local path where you want to save the model
model_dir = rag_config.model_dir

# Download and load the model from huggingface
print("Downloading the model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded and loaded successfully!")
os.makedirs(model_dir, exist_ok=True)
model.save(model_dir)
print("Model saved at:", model_dir)