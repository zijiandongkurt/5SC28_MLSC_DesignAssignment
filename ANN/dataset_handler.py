import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class UnbalancedDiskDataset(Dataset):
    """Creates sequential chunks of data for RNN training."""
    def __init__(self, u, y, seq_length):
        self.u = u
        self.y = y
        self.seq_length = seq_length

    def __len__(self):
        # Number of possible sequences of length `seq_length`
        return len(self.u) - self.seq_length + 1

    def __getitem__(self, idx):
        u_seq = self.u[idx : idx + self.seq_length]
        y_seq = self.y[idx : idx + self.seq_length]
        return torch.FloatTensor(u_seq), torch.FloatTensor(y_seq)

def analyze_and_plot_data(df):
    """Plots the raw dataset to visualize the dynamics."""
    print("--- Dataset Analysis ---")
    print(df.describe())
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['u'][:1000], label='Voltage (u)', color='orange')
    plt.title('First 1000 steps of Input Voltage')
    plt.ylabel('Voltage (V)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df['y'][:1000], label='Measured Angle (y)', color='blue')
    plt.title('First 1000 steps of Output Angle')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    print("Saved dataset analysis plot to 'dataset_analysis.png'")
    
    plt.figure()
    plt.boxplot([df['u'], df['y']], labels=['Voltage (u)', 'Angle (y)'])
    plt.title('Boxplot of Voltage and Angle')
    plt.savefig('dataset_boxplot.png')
    print("Saved dataset boxplot to 'dataset_boxplot.png'")

def prepare_dataloaders(filepath, seq_length=100, batch_size=32, train_split=0.6, val_split=0.2):
    # 1. Load Data
    df = pd.read_csv(filepath)
    
    df.columns = ['u', 'y']
    
    # Extract numpy arrays and reshape for PyTorch (N, 1)
    u_data = df['u'].values.reshape(-1, 1)
    y_data = df['y'].values.reshape(-1, 1)
    
    # 3. Optional but recommended: Scale the data for stable training
    u_scaler = StandardScaler().fit(u_data)
    y_scaler = StandardScaler().fit(y_data)
    
    u_scaled = u_scaler.transform(u_data)
    y_scaled = y_scaler.transform(y_data)

    # 4. Sequential Splitting (DO NOT shuffle randomly for time-series)
    n_total = len(df)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    u_train, y_train = u_scaled[:n_train], y_scaled[:n_train]
    u_val, y_val = u_scaled[n_train : n_train+n_val], y_scaled[n_train : n_train+n_val]
    u_test, y_test = u_scaled[n_train+n_val:], y_scaled[n_train+n_val:]
    
    # 5. Create PyTorch Datasets
    train_dataset = UnbalancedDiskDataset(u_train, y_train, seq_length)
    val_dataset = UnbalancedDiskDataset(u_val, y_val, seq_length)
    test_dataset = UnbalancedDiskDataset(u_test, y_test, seq_length)
    
    # 6. Create DataLoaders
    # We shuffle the STARTING INDEX of the chunks, which is perfectly safe and helps training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, u_scaler, y_scaler, df