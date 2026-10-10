import os

import numpy as np
import torch
import torch.nn as nn


# Re-define the exact architecture from the training script.
class DissonanceScorer(nn.Module):
    def __init__(self, input_dim=768):
        super(DissonanceScorer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()


_model = None
_load_attempted = False
_model_path = os.path.join(os.path.dirname(__file__), "models_cache", "dissonance_scorer_weights.pth")


def _load_model():
    """Load the trained scorer once per worker process."""
    global _model, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True

    if not os.path.exists(_model_path):
        print("[Dissonance] Weights not found. Scores will stay at 0.0 until the model is trained.")
        return None

    try:
        device = torch.device("cpu")
        model = DissonanceScorer().to(device)
        state = torch.load(_model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        _model = model
        print("[Dissonance] Loaded audio-visual dissonance scorer.")
    except Exception as exc:
        print(f"[Dissonance] Could not load weights, continuing without scores: {exc}")
        _model = None
    return _model


def warm_dissonance_model():
    """Call before a pipeline pass so the first frame does not pay the load cost."""
    return _load_model()


def get_dissonance_score(audio_embedding: np.ndarray, visual_embedding: np.ndarray) -> float:
    """Score the semantic gap between one frame's audio and visual embeddings."""
    model = _load_model()
    if model is None:
        return 0.0

    try:
        audio_vec = np.asarray(audio_embedding, dtype=np.float32).reshape(-1)
        visual_vec = np.asarray(visual_embedding, dtype=np.float32).reshape(-1)
        combined = np.concatenate((audio_vec, visual_vec))
        expected = model.network[0].in_features
        if combined.shape[-1] != expected:
            print(
                f"[Dissonance] Embedding width {combined.shape[-1]} does not match "
                f"the scorer input {expected}. Skipping this frame."
            )
            return 0.0

        device = next(model.parameters()).device
        with torch.no_grad():
            tensor_input = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device)
            score = model(tensor_input).item()
        return round(float(score), 4)
    except Exception as exc:
        print(f"[Dissonance] Scoring failed for this frame: {exc}")
        return 0.0
