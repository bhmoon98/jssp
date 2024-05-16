import torch
import torch.nn as nn

import torch
import torch.nn as nn

class JSSPEmbedding(nn.Module):
    def __init__(self, job_count, machine_count, d_model):
        super(JSSPEmbedding, self).__init__()
        # machine_count = max_operations
        self.d_model = d_model
        # Categorical data embeddings
        self.job_embedding = nn.Embedding(job_count, d_model)
        self.machine_embedding = nn.Embedding(machine_count, d_model)
        self.sequence_embedding = nn.Embedding(machine_count, d_model)
        # Continuous data embeddings
        self.time_embedding = nn.Linear(1, d_model)
        self.final_projection = nn.Linear(4 * d_model, d_model)
    
    def forward(self, x):
        batch_size, num_jobs, num_operations, _ = x.shape  # (32, 100, 20, 2)
        times = x[:, :, :, 0]  # (32, 100, 20)
        machines = x[:, :, :, 1]  # (32, 100, 20)
        
        # Generate job and sequence indices
        job_indices = torch.arange(num_jobs).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, num_operations).to(x.device)
        sequence_indices = torch.arange(num_operations).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_jobs, 1).to(x.device)
        # job_indices shape: (batch_size, num_jobs, num_operations) -> (32, 100, 20)
        # sequence_indices shape: (batch_size, num_jobs, num_operations) -> (32, 100, 20)
        
        # Flatten for embedding lookup
        job_indices = job_indices.flatten()  # (batch_size * num_jobs * num_operations) -> (32 * 100 * 20) = (64000)
        machine_indices = machines.flatten()  # (batch_size * num_jobs * num_operations) -> (32 * 100 * 20) = (64000)
        sequence_indices = sequence_indices.flatten()  # (batch_size * num_jobs * num_operations) -> (32 * 100 * 20) = (64000)
        time_values = times.flatten().unsqueeze(-1).float().to(x.device)  # (batch_size * num_jobs * num_operations, 1) -> (64000, 1)
        
        # Embeddings
        job_emb = self.job_embedding(job_indices)  # Shape: (batch_size * num_jobs * num_operations, d_model)
        machine_emb = self.machine_embedding(machine_indices)  # Shape: (batch_size * num_jobs * num_operations, d_model)
        sequence_emb = self.sequence_embedding(sequence_indices)  # Shape: (batch_size * num_jobs * num_operations, d_model)
        time_emb = self.time_embedding(time_values)  # Shape: (batch_size * num_jobs * num_operations, d_model)
        
        # Concatenate embeddings
        concat_emb = torch.cat((job_emb, machine_emb, sequence_emb, time_emb), dim=-1)  # Shape: (batch_size * num_jobs * num_operations, 4 * d_model)
        
        # Project to final embedding size
        final_emb = self.final_projection(concat_emb)  # Shape: (batch_size * num_jobs * num_operations, d_model)
        
        # Reshape to (batch_size, num_jobs * num_operations, d_model)
        final_emb = final_emb.view(batch_size, num_jobs * num_operations, self.d_model)
        
        return final_emb

class JSSPTransformer(nn.Module):
    def __init__(self, job_count, machine_count, d_model, num_heads, num_layers, num_heuristics=8):
        super(JSSPTransformer, self).__init__()
        
        # Embedding layer
        self.embedding_layer = JSSPEmbedding(job_count, machine_count, d_model)
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Heuristic prediction layer
        self.heuristic_predictor = nn.Linear(d_model, num_heuristics)
    
    def forward(self, x):
        # Embedding
        embeddings = self.embedding_layer(x)  # Shape: (batch_size, num_jobs * num_operations, d_model)
        
        # Transformer encoder expects input of shape (seq_len, batch_size, d_model)
        embeddings = embeddings.transpose(0, 1)  # Shape: (num_jobs * num_operations, batch_size, d_model)
        
        # Transformer encoder
        x = self.transformer_encoder(embeddings)  # Shape: (num_jobs * num_operations, batch_size, d_model)
        
        # Pooling
        x = x.mean(dim=0)  # Shape: (batch_size, d_model)
        
        # Predict the heuristic
        x = self.heuristic_predictor(x)  # Shape: (batch_size, num_heuristics)
        
        return x