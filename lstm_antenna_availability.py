import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import wandb

# Multivariate Parallel LSTM Time Series Sequence Classification Model

'''
Model Architecture:
Input size: 4
Hidden size: 64
Number of layers: 2
Linear Layer: maps the LSTM's output to the final output size of 4
Softmax Layer: convert the output into probabilities for each antenna
'''

# TODO: tune the hyperparameters; test on data and compare with no switching penalty

class AntennaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AntennaLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

def create_sequences(data, n_past, n_future):
    X, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        X.append(data[i:(i + n_past + n_future)])
        y.append(np.argmax(data[i + n_past - 1]))
    return np.array(X), np.array(y)

def train_model(model, X_train, y_train, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        wandb.log({"epoch": epoch + 1, "loss": loss.item()})

def predict_antenna(model, X):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Hyperparameters
#Changes from last
'''
V1:
n_past = 6
n_future = 6
epochs = 1000
num_layers = 2
hidden_size = 64
'''

input_size = 4  # 4 antennas
hidden_size = 256
num_layers = 4
output_size = 4
n_past = 25
n_future = 25
epochs = 600
learning_rate = 0.0001

wandb.init(project="antenna_prioritization", config={
    "batch_size": 16,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "model_type": "LSTM"
})

data = pd.read_csv("./data/hsdata_altered.csv")

del data["Unnamed: 16"]
del data["Unnamed: 17"]

data = data[[f"{antenna}_link_budget" for antenna in ["DS24", "DS34", "DS54", "WPSA"]]].values

X, y = create_sequences(data, n_past, n_future)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

model = AntennaLSTM(input_size, hidden_size, num_layers, output_size)
wandb.watch(model, log="all")
train_model(model, X, y, epochs, learning_rate)

model_path = "antenna_prioritization_lstm.pth"
torch.save(model.state_dict(), model_path)
wandb.save(model_path)

def prioritize_antenna(model, X, previous_antenna, switching_penalty=0.25):
    outputs = model(X.unsqueeze(0))
    probabilities = outputs.squeeze().numpy()
    
    if previous_antenna is not None:
        probabilities[previous_antenna] -= switching_penalty
    
    return np.argmax(probabilities)

#PREDICTION AND SWITCHING PENALTY
previous_antenna = None
selected_antennas= []
for i in range(len(X)):
    sample = X[i]
    selected_antenna = prioritize_antenna(model, sample, previous_antenna)
    selected_antennas.append(selected_antenna)
    # print(f"Time step {i+1}: Selected antenna {selected_antenna + 1}")
    previous_antenna = selected_antenna
df = pd.DataFrame(selected_antennas, columns=["Time (min)","Available Satellite"])
data["MISSION ELAPSED TIME (min)"] = data["Time (min)"]
df["Available Satellite"] = selected_antennas
df.to_csv("antenna_availability_lstm.csv", index=False)