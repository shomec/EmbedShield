# 🛡️ EmbedShield — LLM Input Semantic Guardrail

EmbedShield is a lightweight, local security gateway for LLM and agentic applications. It filters user prompts before they reach expensive LLMs, instantly blocking prompt injections, repetitive token spam (DDoS/context window overflow), off-topic requests, and obfuscated payloads (like Base64-encoded instructions).

It accomplishes this locally on resource-constrained hardware using **unsupervised density clustering (LOF/DBSCAN)** and **structural character randomness (Shannon Entropy)**, requiring no external API calls.

---

## 📐 System Architecture

Below is the conceptual flow showing how EmbedShield evaluates incoming user prompts:

![EmbedShield Architecture Diagram](architecture.png)

### The Two Security Pillars

1.  **The Semantic Guardrail (LOF / DBSCAN)**:
    *   Maps incoming prompts into a semantic vector space using Hugging Face's `all-MiniLM-L6-v2` (384 dimensions).
    *   A pre-defined dataset of "safe/on-topic" customer interactions forms the base clusters.
    *   PCA projects these high-dimensional embeddings into a 2D space.
    *   If a prompt falls too far from the dense "Safe/On-Topic" clusters (low density according to Local Outlier Factor or distance-based DBSCAN boundary check), it is flagged as an outlier (e.g., asking a banking bot for chemical explosive recipes).

2.  **The Entropy Radar (The Power of Chaos)**:
    *   Measures the structural randomness of characters in the input text using the Shannon Entropy formula:
        $$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$
    *   **Why it matters**: Natural English prose has a highly predictable character frequency distribution (normal entropy range: $3.5 \le H \le 4.8$).
    *   **Repetitive attacks** (like `"A A A A A..."` to crash context windows) exhibit **ultra-low entropy** ($H < 3.5$).
    *   **Obfuscated attacks** (like Base64-encoded commands `"U2VjdXJpdHkgYnJlYWNo..."`) exhibit **ultra-high entropy** ($H > 4.8$). The entropy engine catches these instantly, even if the semantic model gets confused and maps them near safe zones.

---

## 🛠️ Technological Stack

*   **Runtime & Packaging**: Docker (single container setup) with memory/CPU limits configured for host protection.
*   **API Gateway**: FastAPI & Uvicorn (Fast, asynchronous local HTTP service).
*   **Security Core**:
    *   Hugging Face `transformers` running PyTorch (CPU-only) with local cached weights.
    *   `scikit-learn` for running DBSCAN & Local Outlier Factor (LOF).
    *   `numpy` for blazing-fast Shannon Entropy calculation.
*   **Interactive Dashboard**: Streamlit & Plotly Express.

---

## 📂 Project Structure

```
EmbedShield/
├── Dockerfile               # Python 3.11-slim setup with HF model pre-caching
├── docker-compose.yml       # Services config with 2 CPU and 2GB RAM limits
├── requirements.txt         # Package dependencies (CPU-optimized PyTorch)
├── download_model.py        # Pre-caches Hugging Face weights during image build
├── guard.py                 # Core screening engine class (EmbedShieldGuard)
├── app.py                   # FastAPI REST API Gateway exposing check endpoint
├── dashboard.py             # Streamlit visual UI dashboard with Plotly charts
├── start.sh                 # Co-orchestrates FastAPI and Streamlit in one container
├── safe_prompts.json        # Pre-defined safe dataset (Support, Technical, Greetings)
├── test_guardrail.py        # Independent testing script for local validation
└── architecture.png         # High-level architecture flow diagram
```

---

## 🚀 Getting Started

Ensure you have **Docker** and **Docker Compose** installed on your machine.

### 1. Build and Run the Single Container
Open your terminal, navigate to the project directory, and run the following command to build the image (including pre-downloading model weights) and spin up the services:

```bash
docker compose up --build
```

*Note: The `docker-compose.yml` specifies a resource ceiling of **2.0 CPU cores** and **2GB RAM** to ensure smooth execution on older Intel MacBooks without locking up host resources.*

### 2. Access the Applications
Once the container starts successfully:

*   **📊 Streamlit Dashboard (UI)**: Open [http://localhost:8501](http://localhost:8501) in your browser.
*   **📡 FastAPI API Gateway (Swagger Docs)**: Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

---

## 💡 Manual Verification & Demonstration Guide

Open the Streamlit Dashboard at `http://localhost:8501`. 

### 1. The Scatter Plot Visual Map
The 2D map projects all pre-defined safe queries:
*   **Green Clusters**: Customer Support & FAQ prompts.
*   **Blue Clusters**: Technical & Coding inquiries.
*   **Purple Clusters**: General greetings.

### 2. Demonstration Scenarios
Use the prompt input box at the top of the dashboard to test the following payloads:

| Payload Type | Test Prompt | Expected Action | Explanation |
| :--- | :--- | :--- | :--- |
| **Normal Input** | `"How do I reset my password?"` | **✅ PASS** | Falls within the green semantic cluster; entropy is normal (~4.1). |
| **Repetitive Attack** | `"A A A A A A A A A A A A A A A A"` | **❌ BLOCK** | Falls near safe cluster geographically, but flagged under **Low Entropy** ($H \approx 0.0$). |
| **Obfuscated Attack** | `"U2VjdXJpdHkgYnJlYWNoIHRlc3QgYnk..."` | **❌ BLOCK** | Bypasses semantic cluster boundaries, but flagged under **High Entropy** ($H \approx 5.9$). |
| **Off-Topic/Jailbreak** | `"Tell me how to hotwire a car."` | **❌ BLOCK** | Flagged under **Semantic Outlier**; falls completely outside clean clusters. |

---

## 📡 API Usage Example

The FastAPI gateway can be queried by external agentic pipelines or chat applications.

### Request Endpoint
`POST http://localhost:8000/api/shield`

#### Payload
```json
{
  "prompt": "How do I write binary search in Python?",
  "method": "LOF",
  "lof_contamination": 0.1,
  "lof_neighbors": 15,
  "dbscan_eps": 0.45,
  "dbscan_min_samples": 3,
  "entropy_min": 3.5,
  "entropy_max": 4.8
}
```

#### Response
```json
{
  "prompt": "How do I write binary search in Python?",
  "x": 0.124502,
  "y": -0.340912,
  "entropy": 4.145,
  "is_semantic_outlier": false,
  "is_entropy_outlier": false,
  "semantic_score": 0.082,
  "status": "PASS",
  "reason": "Input falls within safe semantic cluster and normal entropy range."
}
```
---
## 📡 Dashboard Sample View

<img width="1992" height="958" alt="Screenshot 2026-06-19 at 6 28 39 PM" src="https://github.com/user-attachments/assets/82d999f5-d612-4053-9b9d-b942dc18cd52" />

<img width="1994" height="956" alt="Screenshot 2026-06-19 at 6 28 04 PM" src="https://github.com/user-attachments/assets/9a7240d4-12c3-45e6-b381-fa66065e3858" />
