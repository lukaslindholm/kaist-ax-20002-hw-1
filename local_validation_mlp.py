import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASK = 'D'

# 1. DATAFÖRBEREDELSE
print(f"--- Tränar PyTorch MLP för Task {TASK} ---")
train = pd.read_csv(f'data/train_{TASK}.csv')
y = train['y'].to_numpy(dtype=float)
X_df = pd.get_dummies(train.drop(columns='y'), drop_first=True, dtype=float)

# Vi kan behålla log-transformen från tidigare insikter
X_df['x07_log'] = np.log1p(X_df['x07'])
X_df['x09_is_max'] = (X_df['x09'] > 45).astype(float)

X = X_df.to_numpy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisering är KRITISKT för neurala nätverk
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Konvertera till PyTorch-tensorer
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Skapa DataLoaders för mini-batch träning
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 2. DEFINIERA NÄTVERKET (MLP)
class QuantileMLP(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(QuantileMLP, self).__init__()
        # Ett brett nätverk med två dolda lager för att fånga komplexitet
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

model = QuantileMLP(input_dim=X_train.shape[1], out_dim=len(QUANTILES))
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 3. DEFINIERA PINBALL LOSS FÖR PYTORCH
def pinball_loss_torch(y_true, y_pred, quantiles):
    loss = 0
    for i, q in enumerate(quantiles):
        error = y_true - y_pred[:, i].view(-1, 1)
        loss += torch.max(q * error, (q - 1) * error).mean()
    return loss / len(quantiles)

# 4. TRÄNINGSLOOP
epochs = 150
print("Startar träning...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = pinball_loss_torch(batch_y, preds, QUANTILES)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch + 1) % 25 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = pinball_loss_torch(y_val_t, val_preds, QUANTILES)
        print(f"Epoch {epoch+1}/{epochs} | Tränings-Loss: {total_loss/len(train_loader):.4f} | Val-Loss: {val_loss.item():.4f}")

# 5. UTVÄRDERING OCH D2 SCORE
model.eval()
with torch.no_grad():
    task_preds = model(X_val_t).numpy()

# Klippning baserat på domänkunskap (samma som för LightGBM)
task_preds = np.clip(task_preds, 0.0, 20.0)
task_preds = np.sort(task_preds, axis=1)

# Räkna ut Numpy Pinball Loss för att jämföra rättvist med din baslinje
def pinball_loss_np(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

# Baseline loss
baseline_q_values = np.quantile(y_train, QUANTILES)
total_baseline_loss = 0
for j, q in enumerate(QUANTILES):
    baseline_preds = np.full_like(y_val, baseline_q_values[j])
    total_baseline_loss += pinball_loss_np(y_val, baseline_preds, q)
avg_baseline_loss = total_baseline_loss / len(QUANTILES)

# Modellens loss
total_my_loss = 0
for j, q in enumerate(QUANTILES):
    total_my_loss += pinball_loss_np(y_val, task_preds[:, j], q)
avg_my_loss = total_my_loss / len(QUANTILES)

d2_score = 1 - (avg_my_loss / avg_baseline_loss)
print(f"\nBaseline PL: {avg_baseline_loss:.4f} | MLP PL: {avg_my_loss:.4f}")
print(f"--> Lokal D2 Score för Task D (PyTorch): {d2_score:.4f} <--")