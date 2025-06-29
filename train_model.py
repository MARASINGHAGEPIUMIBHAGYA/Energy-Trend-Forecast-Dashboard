import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Paths
os.makedirs("model", exist_ok=True)
MODEL_PATH = "model/lstm_model.pth"
SCALER_PATH = "model/scaler.pkl"
SAMPLE_VALUES_PATH = "model/sample_values.pkl"

# Load and preprocess data
df = pd.read_csv("data/household_power_consumption.txt", sep=';', na_values='?', low_memory=False)
df['ds'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df = df[['ds', 'Global_active_power']].rename(columns={'Global_active_power': 'y'})
df['y'] = pd.to_numeric(df['y'], errors='coerce').interpolate()
df = df.set_index('ds').resample('H').mean().reset_index()
df = df[df['y'] > 0.1]
df['y'] = df['y'].rolling(window=3, min_periods=1).mean()

# Normalize
scaler = MinMaxScaler()
scaled_y = scaler.fit_transform(df[['y']])

# Sequence generation
SEQ_LENGTH = 24
X, y = [], []
for i in range(len(scaled_y) - SEQ_LENGTH - 1):
    X.append(scaled_y[i:i+SEQ_LENGTH])
    y.append(scaled_y[i+SEQ_LENGTH])

X, y = np.array(X), np.array(y)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Define single-output LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = LSTMModel()

# Training
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = loss_fn(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_loss = loss_fn(val_preds, y_val_tensor)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

# Save everything
torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
sample_values = scaler.inverse_transform(X_train[0].reshape(-1, 1)).flatten().tolist()
joblib.dump(sample_values, SAMPLE_VALUES_PATH)

print("✅ Training complete and model saved.")
