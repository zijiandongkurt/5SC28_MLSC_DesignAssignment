import torch
import matplotlib.pyplot as plt
import os
import json
import datetime
from dataset_handler import prepare_dataloaders, analyze_and_plot_data
from model import NOE_GRU
from trainer import train_model

def main():
    # 1. Get the directory where main.py is currently located (.../ANN)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # --- NEW: Create a results folder inside the ANN directory ---
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True) # Creates it safely if it doesn't exist
    
    FILE_PATH = os.path.join(
        BASE_DIR, 
        '..', 
        'gym-unbalanced-disk-master', 
        'disc-benchmark-files', 
        'training-val-test-data.csv'
    )
    
    # --- Configuration / Hyperparameters ---
    SEQ_LENGTH = 100       
    BATCH_SIZE = 64
    HIDDEN_DIM = 16        
    LEARNING_RATE = 0.005
    NUM_EPOCHS = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Loading and analyzing dataset...")
    train_loader, val_loader, test_loader, u_scaler, y_scaler, df = prepare_dataloaders(
        filepath=FILE_PATH,
        seq_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE
    )

    # 2. Analyze (Called only in main script, not in sweeps)
    analyze_and_plot_data(df)

    # 2. Initialize Model
    print("Initializing NOE GRU Model...")
    model = NOE_GRU(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1)

    # --- Dynamic Naming Setup ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_filename = f"NOE_GRU_{HIDDEN_DIM}units_{timestamp}"
    
    # --- NEW: Define the path for the best model to be saved inside /results/ ---
    model_save_path = os.path.join(RESULTS_DIR, f"best_model_{base_filename}.pth")

    # 3. Train Model
    print("Starting Training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        save_path=model_save_path # <--- Passed the new path to the trainer
    )

    # 4. Save raw data to JSON (inside /results/)
    data_to_save = {
        "model_type": "NOE_GRU",
        "hidden_dim": HIDDEN_DIM,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    
    json_path = os.path.join(RESULTS_DIR, f"learning_curve_{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"\nSaved raw learning curves to '{json_path}'")

    # 5. Plot Training Results (inside /results/)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'NOE GRU Training Curve ({base_filename})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_DIR, f"plot_{base_filename}.png")
    plt.savefig(plot_path)
    print(f"Saved training curve plot to '{plot_path}'")

if __name__ == "__main__":
    main()