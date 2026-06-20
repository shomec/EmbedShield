import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from collections import Counter
import os

class EmbedShieldGuard:
    def __init__(self, safe_prompts_path: str):
        self.safe_prompts_path = safe_prompts_path
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Load Hugging Face tokenizer and model from local cache
        print(f"Loading Hugging Face model and tokenizer from '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        
        # Default Entropy thresholds (Normal English range)
        self.entropy_min = 3.5
        self.entropy_max = 4.8
        
        # Load and pre-embed the dataset
        self.load_safe_prompts()
        self.compute_safe_embeddings()
        
        # Fit PCA to project from 384D to 2D
        self.fit_pca()
        
        # Initialize default LOF and DBSCAN parameters and fit
        self.update_boundaries(
            lof_contamination=0.1,
            lof_neighbors=15,
            dbscan_eps=0.45,
            dbscan_min_samples=3
        )

    def load_safe_prompts(self):
        """Loads pre-defined safe dataset from a JSON file."""
        if not os.path.exists(self.safe_prompts_path):
            raise FileNotFoundError(f"Safe prompts file not found at: {self.safe_prompts_path}")
            
        with open(self.safe_prompts_path, "r") as f:
            self.safe_prompts = json.load(f)
            
        # Calculate entropy for all training samples
        for item in self.safe_prompts:
            item["entropy"] = self.calculate_entropy(item["prompt"])

    def compute_safe_embeddings(self):
        """Generates L2-normalized embeddings for all safe prompts using all-MiniLM-L6-v2."""
        print(f"Embedding {len(self.safe_prompts)} safe prompts...")
        self.safe_embeddings = np.array([self.embed_text(item["prompt"]) for item in self.safe_prompts])

    def fit_pca(self):
        """Fits PCA on the high-dimensional embeddings to map them to 2D space."""
        # all-MiniLM-L6-v2 has 384 dimensions
        self.pca = PCA(n_components=2, random_state=42)
        self.safe_coords_2d = self.pca.fit_transform(self.safe_embeddings)
        
        # Store coordinates on the prompt objects for visualization
        for i, item in enumerate(self.safe_prompts):
            item["x"] = float(self.safe_coords_2d[i, 0])
            item["y"] = float(self.safe_coords_2d[i, 1])

    def embed_text(self, text: str) -> np.ndarray:
        """Tokenizes text, runs through the transformer, performs mean-pooling, and L2-normalizes."""
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean Pooling - Take attention mask into account for correct averaging
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        # L2 normalization to project vectors onto a unit sphere (essential for cosine similarity)
        embedding_norm = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)
        return embedding_norm[0].numpy()

    def calculate_entropy(self, text: str) -> float:
        """Computes the Shannon Entropy of the character distribution in the text using numpy."""
        if not text:
            return 0.0
        
        char_counts = Counter(text)
        total_chars = len(text)
        
        # Probabilities of each character
        probs = np.array([count / total_chars for count in char_counts.values()])
        
        # Shannon Entropy Formula: H(X) = -sum( P(x) * log2(P(x)) )
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    def update_boundaries(self, lof_contamination=0.1, lof_neighbors=15, dbscan_eps=0.45, dbscan_min_samples=3):
        """Re-fits LOF and DBSCAN with the given configuration parameters."""
        # 1. Local Outlier Factor (Novelty = True allows predicting on new points)
        self.lof = LocalOutlierFactor(
            n_neighbors=lof_neighbors, 
            novelty=True, 
            contamination=lof_contamination, 
            metric="euclidean"
        )
        self.lof.fit(self.safe_embeddings)
        
        # 2. DBSCAN Clustering (assigns cluster IDs to existing data points)
        self.dbscan = DBSCAN(
            eps=dbscan_eps, 
            min_samples=dbscan_min_samples, 
            metric="euclidean"
        )
        self.dbscan.fit(self.safe_embeddings)
        
        # Update training data labels (0, 1, 2, ... are clusters, -1 is noise/outlier)
        self.safe_labels = self.dbscan.labels_
        for i, item in enumerate(self.safe_prompts):
            item["cluster"] = int(self.safe_labels[i])

    def evaluate_prompt(self, prompt: str, method: str = "LOF", lof_contamination=0.1, lof_neighbors=15, dbscan_eps=0.45, dbscan_min_samples=3) -> dict:
        """
        Processes a live prompt.
        1. Generates 384D embedding.
        2. Projects embedding to 2D coordinates.
        3. Evaluates semantic density outlier status (DBSCAN distance or LOF score).
        4. Calculates Shannon Entropy.
        5. Returns combined security decisions.
        """
        # Get embedding and project to 2D
        emb = self.embed_text(prompt)
        coord_2d = self.pca.transform([emb])[0]
        x_2d, y_2d = float(coord_2d[0]), float(coord_2d[1])
        
        # Compute structural metrics
        entropy = self.calculate_entropy(prompt)
        
        is_semantic_outlier = False
        semantic_score = 0.0
        
        if method == "LOF":
            # Re-fit LOF with current sliders (very fast on small datasets)
            lof = LocalOutlierFactor(
                n_neighbors=lof_neighbors, 
                novelty=True, 
                contamination=lof_contamination, 
                metric="euclidean"
            )
            lof.fit(self.safe_embeddings)
            
            # Predict outlier status: 1 = inlier, -1 = outlier
            pred = lof.predict([emb])[0]
            is_semantic_outlier = (pred == -1)
            
            # Negative outlier factor (higher means more outlier)
            semantic_score = float(-lof.decision_function([emb])[0])
        else:  # DBSCAN Boundary
            dbscan = DBSCAN(
                eps=dbscan_eps, 
                min_samples=dbscan_min_samples, 
                metric="euclidean"
            )
            dbscan.fit(self.safe_embeddings)
            
            # Find core samples to determine boundary distance
            core_indices = dbscan.core_sample_indices_
            if len(core_indices) > 0:
                core_embeddings = self.safe_embeddings[core_indices]
                distances = np.linalg.norm(core_embeddings - emb, axis=1)
                min_distance = np.min(distances)
                is_semantic_outlier = min_distance > dbscan_eps
                semantic_score = float(min_distance)
            else:
                # Fallback to all safe embeddings if parameters produce no core points
                distances = np.linalg.norm(self.safe_embeddings - emb, axis=1)
                min_distance = np.min(distances)
                is_semantic_outlier = min_distance > dbscan_eps
                semantic_score = float(min_distance)
                
        # Evaluate entropy bounds
        is_entropy_outlier = (entropy < self.entropy_min) or (entropy > self.entropy_max)
        
        # Combine Pillars: Either path can trigger a Block
        status = "BLOCK" if (is_semantic_outlier or is_entropy_outlier) else "PASS"
        
        # Explain why
        reasons = []
        if is_semantic_outlier:
            reasons.append(f"Semantic Outlier (Distance/Score: {semantic_score:.3f})")
        if is_entropy_outlier:
            if entropy < self.entropy_min:
                reasons.append(f"Anomalous Low Entropy (H: {entropy:.3f} < {self.entropy_min}) - High repetition, DDoS/Noise pattern")
            else:
                reasons.append(f"Anomalous High Entropy (H: {entropy:.3f} > {self.entropy_max}) - Obfuscated/Base64/Fuzzing payload")
                
        reason = " & ".join(reasons) if reasons else "Input falls within safe semantic cluster and normal entropy range."
        
        return {
            "prompt": prompt,
            "x": x_2d,
            "y": y_2d,
            "entropy": entropy,
            "is_semantic_outlier": bool(is_semantic_outlier),
            "is_entropy_outlier": bool(is_entropy_outlier),
            "semantic_score": semantic_score,
            "status": status,
            "reason": reason
        }
