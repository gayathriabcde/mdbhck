import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlibbb.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc


# ==========================================
# 1. Recreate the Model Architecture
# ==========================================
class DissonanceScorer(nn.Module):
    def __init__(self, input_dim=768):
        super(DissonanceScorer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # <-- Added missing Dropout
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # <-- Added missing Dropout
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()


# ==========================================
# 2. Helper: Mathematical Baseline
# ==========================================
def calculate_cosine_baseline(emb_a, emb_b):
    """The Zero-Shot Baseline: 1 - Cosine Similarity"""
    dot_product = np.sum(emb_a * emb_b, axis=1)
    norm_a = np.linalg.norm(emb_a, axis=1)
    norm_b = np.linalg.norm(emb_b, axis=1)
    similarity = dot_product / (norm_a * norm_b)
    # Dissonance is the opposite of similarity
    return 1 - similarity


# ==========================================
# 3. Main Evaluation Function
# ==========================================
def evaluate():
    print("📊 Loading Data and Models...")
    df = pd.read_csv("resources/dataset_prelabeled.csv")

    # ---> NEW FIX: Drop empty rows and force labels to be integers <---
    df = df.dropna(subset=['is_dissonant', 'audio_transcript', 'visual_caption'])
    df['is_dissonant'] = df['is_dissonant'].astype(int)

    # Optional: If your dataset is huge, sample 500 rows for faster local testing
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    true_labels = df['is_dissonant'].values

    # Load Embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    audio_embs = embedder.encode(df['audio_transcript'].tolist())
    visual_embs = embedder.encode(df['visual_caption'].tolist())

    # --- Generate Baseline Predictions ---
    print("🧮 Calculating Zero-Shot Baseline...")
    baseline_probs = calculate_cosine_baseline(audio_embs, visual_embs)
    baseline_preds = (baseline_probs >= 0.5).astype(int)

    # --- Generate Custom Model Predictions ---
    print("🧠 Running Custom Trained Model...")
    model_path = "worker/models_cache/dissonance_scorer_weights.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at {model_path}")
        return

    model = DissonanceScorer()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    combined_embs = np.concatenate((audio_embs, visual_embs), axis=1)
    tensor_input = torch.tensor(combined_embs, dtype=torch.float32)

    with torch.no_grad():
        model_probs = model(tensor_input).numpy()
        model_preds = (model_probs >= 0.5).astype(int)

    # ==========================================
    # 4. Generate the Metrics Report
    # ==========================================
    print("\n" + "=" * 50)
    print("📝 EVALUATION REPORT FOR JOURNAL")
    print("=" * 50)

    print("\n--- BASELINE (Raw Cosine Distance) ---")
    print(f"Accuracy: {accuracy_score(true_labels, baseline_preds):.4f}")
    print(classification_report(true_labels, baseline_preds, target_names=["Redundant (0)", "Dissonant (1)"]))

    print("\n--- ODP-RAG CUSTOM MODEL (Cross-Encoder) ---")
    print(f"Accuracy: {accuracy_score(true_labels, model_preds):.4f}")
    print(classification_report(true_labels, model_preds, target_names=["Redundant (0)", "Dissonant (1)"]))

    # ==========================================
    # 5. Plot the ROC Curve
    # ==========================================
    print("\n📈 Generating ROC Curve plot...")
    fpr_base, tpr_base, _ = roc_curve(true_labels, baseline_probs)
    roc_auc_base = auc(fpr_base, tpr_base)

    fpr_model, tpr_model, _ = roc_curve(true_labels, model_probs)
    roc_auc_model = auc(fpr_model, tpr_model)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_base, tpr_base, color='darkorange', lw=2, linestyle='--',
             label=f'Baseline Cosine (AUC = {roc_auc_base:.2f})')
    plt.plot(fpr_model, tpr_model, color='blue', lw=2, label=f'ODP-RAG Trained Model (AUC = {roc_auc_model:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # 50/50 Guess Line

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (False Alarms)')
    plt.ylabel('True Positive Rate (Correct Hits)')
    plt.title('Receiver Operating Characteristic (ROC) - Tri-Modal Dissonance')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save the plot for your paper
    plot_path = "testing/roc_curve_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC Curve saved to {plot_path}")


if __name__ == "__main__":
    evaluate()