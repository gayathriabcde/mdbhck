import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================
# 1. Configuration & Device Setup
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 2e-4


# ==========================================
# 2. The PyTorch Dataset Definition
# ==========================================
class DissonanceDataset(Dataset):
    def __init__(self, df, embedder):
        self.labels = torch.tensor(df['is_dissonant'].values, dtype=torch.float32)

        print("Embedding Audio Transcripts...")
        audio_embeds = embedder.encode(df['audio_transcript'].tolist(), convert_to_tensor=True)

        print("Embedding Visual Captions...")
        visual_embeds = embedder.encode(df['visual_caption'].tolist(), convert_to_tensor=True)

        # Concatenate [Audio, Visual] -> 384 + 384 = 768 dimensions
        self.features = torch.cat((audio_embeds, visual_embeds), dim=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==========================================
# 3. The Dissonance Scorer (MLP) Architecture
# ==========================================
class DissonanceScorer(nn.Module):
    def __init__(self, input_dim=768):
        super(DissonanceScorer, self).__init__()
        # A lightweight Multi-Layer Perceptron
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting on your small dataset
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output a probability between 0.0 and 1.0
        )

    def forward(self, x):
        return self.network(x).squeeze()


# ==========================================
# 4. The Training Execution
# ==========================================
def train_model():
    # Load your labeled data
    df = pd.read_csv("dataset.csv")  # Change path if needed on Kaggle

    # Split into train/validation (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Load the frozen embedding model
    print("Loading SentenceTransformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

    # Create DataLoaders
    print("Preparing Datasets...")
    train_dataset = DissonanceDataset(train_df, embedder)
    val_dataset = DissonanceDataset(val_df, embedder)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = DissonanceScorer().to(DEVICE)
    criterion = nn.BCELoss()  # Binary Cross Entropy for 0/1 labels
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # The Training Loop
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(DEVICE)
                predictions = model(batch_features)
                # Convert probabilities to strict 0 or 1
                binary_preds = (predictions >= 0.5).float().cpu().numpy()
                val_preds.extend(binary_preds)
                val_targets.extend(batch_labels.numpy())

        # Metrics for your Journal Paper
        val_acc = accuracy_score(val_targets, val_preds)
        val_prec = precision_score(val_targets, val_preds, zero_division=0)
        val_rec = recall_score(val_targets, val_preds, zero_division=0)

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

    # Save the lightweight weights for your Flask App
    torch.save(model.state_dict(), "dissonance_scorer_weights.pth")
    print("\n✅ Training Complete. Weights saved to dissonance_scorer_weights.pth")


if __name__ == "__main__":
    train_model()