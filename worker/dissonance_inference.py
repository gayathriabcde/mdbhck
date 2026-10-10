import os
import torch
import torch.nn as nn
import numpy as np


# Re-define the exact architecture from your training script
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


def get_dissonance_score(audio_embedding: np.ndarray, visual_embedding: np.ndarray) -> float:
    """Loads the trained weights and calculates the semantic gap."""
    model_path = os.path.join(os.path.dirname(__file__), "models_cache", "dissonance_scorer_weights.pth")

    # If model isn't trained/downloaded yet, safely fallback to 0.0
    if not os.path.exists(model_path):
        return 0.0

    device = torch.device("cpu")  # Workers run on CPU to save resources
    model = DissonanceScorer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # Concatenate the float32 vectors (must be float32, NOT the int8 database vectors)
        combined = np.concatenate((audio_embedding, visual_embedding))
        tensor_input = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device)
        score = model(tensor_input).item()

    return round(score, 4)