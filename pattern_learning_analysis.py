"""
# Financial Time Series Embedding & Pattern Learning (BTC)

This notebook implements a deep representation learning approach for BTC price and volume data, inspired by the paper "Deep Representation Learning and Analog Forecasting in Financial Time Series".

## Methodology:
1. **Latent Pattern Recognition**: Instead of direct regression, we project time series windows into a compact embedding space.
2. **Multi-Modal Fusion**: We use both **Price** (Close returns) and **Volume** to capture the "intensity" of market moves.
3. **Contrastive Learning**: A self-supervised approach to learn structural invariants in the data.
4. **Analog Forecasting**: Using similarity search (kNN) in the latent space to find historical analogues.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Load Data
data_path = "data/btcusd_1-min_data.csv"
df = pd.read_csv(data_path)

# Convert timestamp to datetime and set as index
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Timestamp', inplace=True)

# Resample to hourly to reduce noise and speed up training for this demo
df_resampled = df.resample('1H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# Feature Engineering: Log Returns and Normalized Volume
df_resampled['Returns'] = np.log(df_resampled['Close'] / df_resampled['Close'].shift(1))
df_resampled['Vol_Norm'] = np.log(df_resampled['Volume'] + 1) # Log to handle volume spikes
df_resampled.dropna(inplace=True)

# Prepare sliding windows
WINDOW_SIZE = 24 # 24 hours
features = df_resampled[['Returns', 'Vol_Norm']].values

def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i+window_size])
    return np.array(windows)

X = create_windows(features, WINDOW_SIZE)
print(f"Dataset shape: {X.shape} (Windows, Window_Size, Features)")

# %%
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, embedding_dim=32):
        super(TimeSeriesEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, window_size, features) -> (batch, features, window_size)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        embedding = self.fc(x)
        return embedding

# Triplet Dataset for Contrastive Learning
class TripletTimeDataset(Dataset):
    def __init__(self, windows):
        self.windows = torch.FloatTensor(windows)
        
    def __len__(self):
        return len(self.windows) - 100 # Leave room for anchors/positives
    
    def __getitem__(self, idx):
        anchor = self.windows[idx]
        # Positive: A window very close in time (high temporal correlation)
        pos_idx = idx + np.random.randint(1, 5)
        positive = self.windows[pos_idx]
        # Negative: A random window from a different time
        neg_idx = np.random.randint(0, len(self.windows))
        while abs(neg_idx - idx) < 50:
            neg_idx = np.random.randint(0, len(self.windows))
        negative = self.windows[neg_idx]
        
        return anchor, positive, negative

dataset = TripletTimeDataset(X)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# %%
# Training Parameters
embedding_dim = 16
model = TimeSeriesEncoder(input_dim=2, hidden_dim=64, embedding_dim=embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.TripletMarginLoss(margin=1.0)

epochs = 5 # Small number for demonstration
print("Starting training...")

for epoch in range(epochs):
    epoch_loss = 0
    for anchor, positive, negative in dataloader:
        optimizer.zero_grad()
        
        anchor_emb = model(anchor)
        pos_emb = model(positive)
        neg_emb = model(negative)
        
        loss = criterion(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

print("Training complete.")

# %%
# Generate Embeddings for all historical data
model.eval()
with torch.no_grad():
    all_windows_tensor = torch.FloatTensor(X)
    embeddings = model(all_windows_tensor).numpy()

print(f"Generated {len(embeddings)} embeddings.")

# Similarity Search (Analog Forecasting)
def find_analogue(query_idx, embeddings, k=3):
    query_vec = embeddings[query_idx].reshape(1, -1)
    
    # Calculate Cosine Similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    
    # Exclude the query window and its immediate neighbors (avoiding autocorrelation)
    similarities[max(0, query_idx-50):min(len(similarities), query_idx+50)] = -1
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices, similarities[top_k_indices]

# Pick a recent window and find its historical analogue
query_idx = len(X) - 1
analogue_indices, scores = find_analogue(query_idx, embeddings)

print(f"Query window index: {query_idx}")
for i, (idx, score) in enumerate(zip(analogue_indices, scores)):
    print(f"Analogue {i+1}: Index {idx}, Date: {df_resampled.index[idx]}, Similarity: {score:.4f}")

# %%
# Visualize Query vs Analogue
plt.figure(figsize=(15, 6))

# Plot Price Returns
plt.subplot(1, 2, 1)
plt.plot(X[query_idx][:, 0], label='Current (Query)', linewidth=3, color='black')
for i, idx in enumerate(analogue_indices):
    plt.plot(X[idx][:, 0], label=f'Analogue {i+1} (idx {idx})', linestyle='--')
plt.title("Price Return Patterns (Query vs Analogues)")
plt.legend()

# Plot Volume Patterns
plt.subplot(1, 2, 2)
plt.plot(X[query_idx][:, 1], label='Current (Query)', linewidth=3, color='black')
for i, idx in enumerate(analogue_indices):
    plt.plot(X[idx][:, 1], label=f'Analogue {i+1}', linestyle='--')
plt.title("Volume Patterns (Query vs Analogues)")
plt.legend()

plt.tight_layout()
plt.show()

# Visualize Latent Space using T-SNE (Regime Detection)
print("Computing T-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings[:2000]) # Subsample for speed

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, c=np.arange(len(embeddings_2d)), cmap='viridis')
plt.colorbar(label='Time progression')
plt.title("Latent Space Visualization (T-SNE)")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.show()