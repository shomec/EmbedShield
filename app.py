from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from guard import EmbedShieldGuard
import os

app = FastAPI(
    title="EmbedShield API Gateway", 
    description="Local Unsupervised Density & Entropy Guardrail for LLM Inputs"
)

# Path to local dataset
SAFE_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "safe_prompts.json")

# Instantiate core security engine
guard = EmbedShieldGuard(SAFE_PROMPTS_PATH)

class ShieldRequest(BaseModel):
    prompt: str = Field(..., description="The raw user prompt to inspect")
    method: str = Field("LOF", description="Semantic density algorithm to use: 'LOF' or 'DBSCAN'")
    lof_contamination: float = Field(0.1, description="LOF contamination rate (fraction of outliers expected in training)")
    lof_neighbors: int = Field(15, description="LOF number of neighbors to evaluate local density")
    dbscan_eps: float = Field(0.45, description="DBSCAN epsilon radius for core samples")
    dbscan_min_samples: int = Field(3, description="DBSCAN min points required to form a cluster core")
    entropy_min: float = Field(3.5, description="Minimum acceptable Shannon entropy threshold")
    entropy_max: float = Field(4.8, description="Maximum acceptable Shannon entropy threshold")

class ShieldResponse(BaseModel):
    prompt: str
    x: float
    y: float
    entropy: float
    is_semantic_outlier: bool
    is_entropy_outlier: bool
    semantic_score: float
    status: str
    reason: str

@app.get("/health")
def health():
    """Checks service health and returns cached model details."""
    return {
        "status": "healthy", 
        "model": guard.model_name,
        "total_safe_prompts": len(guard.safe_prompts)
    }

@app.get("/api/safe-prompts")
def get_safe_prompts():
    """Retrieves the pre-embedded training dataset with coordinates and cluster labels."""
    return guard.safe_prompts

@app.post("/api/shield", response_model=ShieldResponse)
def shield_check(req: ShieldRequest):
    """
    Screens incoming prompts against the Semantic and Entropy guardrails.
    Returns decision status (PASS/BLOCK) and evaluation scores.
    """
    # Apply entropy overrides to guard engine instance
    guard.entropy_min = req.entropy_min
    guard.entropy_max = req.entropy_max
    
    try:
        result = guard.evaluate_prompt(
            prompt=req.prompt,
            method=req.method,
            lof_contamination=req.lof_contamination,
            lof_neighbors=req.lof_neighbors,
            dbscan_eps=req.dbscan_eps,
            dbscan_min_samples=req.dbscan_min_samples
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guardrail evaluation error: {str(e)}")
