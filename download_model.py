import os
from transformers import AutoTokenizer, AutoModel

def download_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Downloading tokenizer and model for '{model_name}'...")
    
    # This will download the files and cache them in the default Hugging Face cache location (~/.cache/huggingface)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print("Tokenizer and Model downloaded successfully and cached!")

if __name__ == "__main__":
    download_model()
