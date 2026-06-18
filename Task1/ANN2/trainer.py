import torch
import torch.nn as nn
from tqdm import tqdm
import time
import datetime
import optuna

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_path='best_noe_gru_model.pth', optuna_trial=None):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') 
    
    model.to(device)
    
    global_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        running_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for u_seq, y_seq in progress_bar:
            u_seq, y_seq = u_seq.to(device), y_seq.to(device)
            y_init = y_seq[:, 0, :] 
            
            optimizer.zero_grad()
            y_pred_seq = model(u_seq, y_init)
            loss = criterion(y_pred_seq, y_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for u_seq, y_seq in val_loader:
                u_seq, y_seq = u_seq.to(device), y_seq.to(device)
                y_init = y_seq[:, 0, :]
                y_pred_seq = model(u_seq, y_init)
                val_loss = criterion(y_pred_seq, y_seq)
                running_val_loss += val_loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # --- Real-World Clock Math ---
        epoch_end_time = time.time()
        epochs_completed = epoch + 1
        epochs_remaining = num_epochs - epochs_completed
        
        # Calculate times in seconds
        total_time_elapsed = epoch_end_time - global_start_time
        avg_time_per_epoch = total_time_elapsed / epochs_completed
        estimated_time_remaining = avg_time_per_epoch * epochs_remaining
        
        # Calculate total estimated mins
        total_estimated_seconds = avg_time_per_epoch * num_epochs
        total_estimated_mins = int(total_estimated_seconds // 60)
        
        # Calculate real-world completion time (Current Time + Remaining Time)
        completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_time_remaining)
        completion_time_str = completion_time.strftime("%H:%M")
        
        # Print Results
        save_msg = " -> ** Model Saved! **" if avg_val_loss < best_val_loss else ""
        
        # Your requested format is printed here!
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}{save_msg}")
        print(f"Estimated time {total_estimated_mins} mins. Completed at {completion_time_str}\n")

        # --- Optuna Reporting and Pruning ---
        if optuna_trial is not None:
            optuna_trial.report(avg_val_loss, epoch)
            
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()
              
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            
    total_duration = str(datetime.timedelta(seconds=int(time.time() - global_start_time)))
    print(f"Training Complete! Total time: {total_duration}")
            
    return train_losses, val_losses