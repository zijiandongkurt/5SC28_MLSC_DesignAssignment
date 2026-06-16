import os
import torch
import optuna
import json
import datetime
import matplotlib.pyplot as plt
from dataset_handler import prepare_dataloaders
from model import NOE_GRU
from trainer import train_model

# 1. Define paths globally so the objective function can access them
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FILE_PATH = os.path.join(
    BASE_DIR, 
    '..', 
    'gym-unbalanced-disk-master', 
    'disc-benchmark-files', 
    'training-val-test-data.csv'
)

def objective(trial):
    """
    This function defines the hyperparameter search space and 
    trains a model for a single 'trial'.
    """
    # --- 1. Define the Hyperparameter Search Space ---
    # Optuna will pick a new combination of these for every trial
    hidden_dim = trial.suggest_categorical('hidden_dim', [8, 16, 32, 64])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    seq_length = trial.suggest_categorical('seq_length', [50, 100, 150])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True) # Log scale for LR
    
    # Keep epochs slightly lower during sweeps to save time
    num_epochs = 15 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Load Data with the trial's batch_size and seq_length ---
    train_loader, val_loader, test_loader, _, _, _ = prepare_dataloaders(
        filepath=FILE_PATH,
        seq_length=seq_length,
        batch_size=batch_size
    )

    # --- 3. Initialize the Model ---
    model = NOE_GRU(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
    trial_save_path = os.path.join(RESULTS_DIR, f"optuna_trial_{trial.number}_model.pth")

    print(f"\n--- Starting Trial {trial.number} ---")
    
    # --- NEW: We use a try-except block to gracefully handle pruned trials ---
    try:
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            save_path=trial_save_path,
            optuna_trial=trial
        )
        best_val_loss = min(val_losses)
        return best_val_loss
        
    except optuna.exceptions.TrialPruned:
        # If the trial was pruned, we just re-raise it so Optuna records it as 'Pruned'
        raise optuna.exceptions.TrialPruned()

def main():
    print("Starting Optuna Hyperparameter Sweep...")
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    # Create an Optuna study object. We want to 'minimize' the validation loss.
    study = optuna.create_study(
        direction="minimize", 
        study_name="NOE_GRU_Sweep",
        pruner=pruner # <--- Attach it to the study
    )
    
    study.optimize(objective, n_trials=20)
    
    # --- Results Logging ---
    print("\n==================================================")
    print("Sweep Complete!")
    print("Best Trial:")
    best_trial = study.best_trial
    
    print(f"  Lowest Validation Loss: {best_trial.value:.6f}")
    print("  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    # Save the best parameters to a JSON file in your results folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    best_params_path = os.path.join(RESULTS_DIR, f"optuna_best_params_{timestamp}.json")
    
    best_data = {
        "best_val_loss": best_trial.value,
        "best_params": best_trial.params
    }
    
    with open(best_params_path, 'w') as f:
        json.dump(best_data, f, indent=4)
        
    print(f"Saved best hyperparameters to '{best_params_path}'")

    # --- 4. Run a Final Training with the Best Parameters ---
    print("\n--- Starting Final Training with Best Hyperparameters ---")
    
    # Prepare Data with best seq_length and batch_size
    train_loader, val_loader, test_loader, u_scaler, y_scaler, _ = prepare_dataloaders(
        filepath=FILE_PATH,
        seq_length=best_trial.params['seq_length'],
        batch_size=best_trial.params['batch_size']
    )

    # Initialize model with best hidden_dim
    final_model = NOE_GRU(input_dim=1, hidden_dim=best_trial.params['hidden_dim'], output_dim=1)
    print("\nFinal Model Architecture:")
    print(final_model)

    final_save_path = os.path.join(RESULTS_DIR, f"final_best_model_{timestamp}.pth")

    # Train for more epochs (e.g. 30) for the final run
    train_losses, val_losses = train_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30, 
        learning_rate=best_trial.params['learning_rate'],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_path=final_save_path
    )

    # --- 5. Save Final Learning Curves and Plot ---
    final_log_data = {
        "best_params": best_trial.params,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "final_epoch_count": 30
    }
    
    final_json_path = os.path.join(RESULTS_DIR, f"final_learning_curve_{timestamp}.json")
    with open(final_json_path, 'w') as f:
        json.dump(final_log_data, f, indent=4)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Final NOE GRU Training Curve (Sweep Result {timestamp})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    final_plot_path = os.path.join(RESULTS_DIR, f"final_plot_{timestamp}.png")
    plt.savefig(final_plot_path)

    print(f"\nFinal training complete. Best model saved to: {final_save_path}")
    print(f"Final training log saved to: {final_json_path}")
    print(f"Final training plot saved to: {final_plot_path}")
    print("==================================================")

if __name__ == "__main__":
    main()