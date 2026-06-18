import torch
import torch.nn as nn

class NOE_GRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
        super(NOE_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        
        # The input to the GRU is the current voltage u(k) AND the past prediction y_hat(k-1)
        self.gru_cell = nn.GRUCell(input_size=(input_dim + output_dim), hidden_size=hidden_dim)
        
        # Maps the internal hidden state to the physical angle prediction
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, u_seq, y_init):
        """
        u_seq: Tensor of shape (Batch, Seq_Length, 1)
        y_init: Tensor of shape (Batch, 1) - The true angle right before the sequence starts
        """
        batch_size, seq_len, _ = u_seq.size()
        device = u_seq.device
        
        y_preds = []
        y_prev = y_init # Initialize the feedback loop
        
        # Initialize GRU hidden state to zeros
        h_k = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        # Explicit simulation loop over the sequence chunk
        for k in range(seq_len):
            u_k = u_seq[:, k, :]
            
            # 1. NOE STRUCTURE: Concatenate u(k) and y_hat(k-1)
            gru_input = torch.cat([u_k, y_prev], dim=-1)
            
            # 2. Update memory state
            h_k = self.gru_cell(gru_input, h_k)
            
            # 3. Predict current angle y_hat(k)
            y_curr = self.output_layer(h_k)
            y_preds.append(y_curr)
            
            # 4. FEEDBACK LOOP: Set current prediction as previous for the next iteration
            y_prev = y_curr
            
        # Stack predictions back into (Batch, Seq_Length, 1)
        return torch.stack(y_preds, dim=1)