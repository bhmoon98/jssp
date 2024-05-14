import torch
import torch.nn as nn

class JSSPEmbedding(nn.Module):
    def __init__(self, job_count, machine_count, max_operations, d_model):
        super(JSSPEmbedding, self).__init__()
        self.job_embedding = nn.Embedding(job_count, d_model)
        self.machine_embedding = nn.Embedding(machine_count, d_model)
        self.sequence_embedding = nn.Embedding(max_operations, d_model)
        self.time_embedding = nn.Linear(1, d_model)
        self.final_projection = nn.Linear(4 * d_model, d_model)
    
    def forward(self, job, machine, sequence, time):
        job_emb = self.job_embedding(job)  # Shape: (num_nodes, d_model)
        machine_emb = self.machine_embedding(machine)  # Shape: (num_nodes, d_model)
        sequence_emb = self.sequence_embedding(sequence)  # Shape: (num_nodes, d_model)
        time_emb = self.time_embedding(time.unsqueeze(-1))  # Shape: (num_nodes, d_model)
        
        # Concatenate embeddings
        concat_emb = torch.cat((job_emb, machine_emb, sequence_emb, time_emb), dim=-1)  # Shape: (num_nodes, 4 * d_model)
        
        # Project to final embedding size
        final_emb = self.final_projection(concat_emb)  # Shape: (num_nodes, d_model)
        
        return final_emb

class JSSPTransformer(nn.Module):
    def __init__(self, job_count, machine_count, max_operations, d_model, num_heads, num_layers, num_heuristics):
        super(JSSPTransformer, self).__init__()
        
        # Embedding layer
        self.embedding_layer = JSSPEmbedding(job_count, machine_count, max_operations, d_model)
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Heuristic prediction layer
        self.heuristic_predictor = nn.Linear(d_model, num_heuristics)
    
    def forward(self, job, machine, sequence, time):
        # Embedding
        embeddings = self.embedding_layer(job, machine, sequence, time)  # Shape: (num_nodes, d_model)
        
        # Transformer encoder
        x = self.transformer_encoder(embeddings.unsqueeze(1))  # Shape: (num_nodes, 1, d_model)
        
        # Pooling
        x = x.mean(dim=0)  # Shape: (1, d_model)
        
        # Predict the heuristic
        x = self.heuristic_predictor(x.squeeze(0))  # Shape: (num_heuristics)
        
        return x

# Parameters
job_count = 3            # Number of unique jobs
machine_count = 5        # Number of unique machines
max_operations = 5       # Maximum number of operations per job
d_model = 64             # Embedding dimension
num_heads = 8            # Number of attention heads
num_layers = 6           # Number of transformer encoder layers
num_heuristics = 6       # Number of heuristics to predict

# Instantiate the model
model = JSSPTransformer(job_count, machine_count, max_operations, d_model, num_heads, num_layers, num_heuristics)

# Example data
operation_time_matrix = torch.tensor([
    [10, 20, 30, 40, 50],
    [30, 30, 20, 10, 20],
    [40, 30, 10, 50, 30]
])

machine_matrix = torch.tensor([
    [1, 2, 3, 4, 5],
    [3, 4, 5, 2, 1],
    [1, 3, 5, 2, 4]
])

# Flatten the matrices and create job and sequence indices
job_indices = torch.tensor([i for i in range(job_count) for _ in range(max_operations)])
machine_indices = machine_matrix.flatten()
sequence_indices = torch.tensor([i for _ in range(job_count) for i in range(max_operations)])
time_values = operation_time_matrix.flatten().float()

# Forward pass
output = model(job_indices, machine_indices, sequence_indices, time_values)

# Output is a vector of length `num_heuristics` with scores for each heuristic
print(output)
