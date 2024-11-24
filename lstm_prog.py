import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import wandb

Pt = 10
Gt = 9
losses = 19.43
nR = 0.55
la = 0.136363636
kb = -228.6
Ts = 22


def convert_range_to_link_budget(antenna, range_values):
    dr = 34 if 'DS' in antenna else 12
    for i in range(len(range_values)):
        range_values[i] = (10**((1/10)*(Pt+Gt-losses+10*math.log10(nR*(math.pi*dr/la)**2)-20*math.log10(4000*math.pi*range_values[i]/la)-10*math.log10(Ts))))/1000
    return np.minimum(range_values, 10000)


class AntennaDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        del self.data["Unnamed: 16"]
        del self.data["Unnamed: 17"]

        # for antenna in ["DS24", "DS34"]:
        #     self.data[f"{antenna}_link_budget"] = convert_range_to_link_budget(antenna, self.data[f"Range {antenna}"])
        # for antenna in ["DS54", "WPSA"]:
        #     self.data[f"{antenna}_link_budget"] = convert_range_to_link_budget(antenna, self.data[f"{antenna} Range"])

        # for antenna in ["DS24", "DS34", "DS54", "WPSA"]:
        #     self.data[f"{antenna}_link_budget"] /= 10000.0

        self.features = self.data[[f"{antenna}_link_budget" for antenna in ["DS24", "DS34", "DS54", "WPSA"]]].values
        self.targets = np.argsort(-self.features, axis=1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)


class AntennaLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=4):
        super(AntennaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.unsqueeze(1)
            targets = targets.squeeze()

            outputs = model(inputs)
            targets=targets.type(torch.FloatTensor) 
            loss = criterion(outputs, targets)
            
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log metrics to WandB
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss / len(dataloader)})

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")


if __name__ == "__main__":
    wandb.init(project="antenna_prioritization", config={
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 200,
        "model_type": "LSTM"
    })

    file_path = "./data/hsdata_altered.csv"
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs

    dataset = AntennaDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = AntennaLSTM()
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, log="all", log_freq=10)
    train_model(model, dataloader, criterion, optimizer, epochs)

    # Save the model and log it to WandB
    model_path = "antenna_prioritization_lstm.pth"
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)
