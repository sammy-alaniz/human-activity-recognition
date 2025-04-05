import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model (int): The dimension of the input embeddings (must be divisible by n_heads).
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate for regularization.
        """
        super(CrossAttentionLayer, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model  # Total dimension of the model
        self.n_heads = n_heads  # Number of parallel attention heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear layers to project inputs into queries, keys, and values
        self.W_q = nn.Linear(d_model, d_model)  # Query projection (e.g., from language model)
        self.W_k = nn.Linear(d_model, d_model)  # Key projection (e.g., from image encoder)
        self.W_v = nn.Linear(d_model, d_model)  # Value projection (e.g., from image encoder)
        self.W_o = nn.Linear(d_model, d_model)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))  # Scaling factor for stability

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q (torch.Tensor): Query tensor from one source (e.g., language model), shape [batch_size, seq_len_q, d_model]
            K (torch.Tensor): Key tensor from another source (e.g., image encoder), shape [batch_size, seq_len_kv, d_model]
            V (torch.Tensor): Value tensor from the same source as K, shape [batch_size, seq_len_kv, d_model]
            mask (torch.Tensor, optional): Attention mask to ignore certain positions, shape [batch_size, n_heads, seq_len_q, seq_len_kv]
        
        Returns:
            torch.Tensor: Output after cross-attention, shape [batch_size, seq_len_q, d_model]
        """
        batch_size = Q.size(0)
        
        # Linear projections for Q, K, V
        Q = self.W_q(Q)  # [batch_size, seq_len_q, d_model]
        K = self.W_k(K)  # [batch_size, seq_len_kv, d_model]
        V = self.W_v(V)  # [batch_size, seq_len_kv, d_model]
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len_q, d_k]
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len_kv, d_k]
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len_kv, d_k]
        
        # Compute attention scores: (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, n_heads, seq_len_q, seq_len_kv]
        
        # Apply mask (if provided) to prevent attending to certain positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # Replace masked positions with a large negative value
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, n_heads, seq_len_q, seq_len_kv]
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum of values
        context = torch.matmul(attn_weights, V)  # [batch_size, n_heads, seq_len_q, d_k]
        
        # Reshape and project back to original dimension
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len_q, d_model]
        output = self.W_o(context)  # [batch_size, seq_len_q, d_model]
        
        return output
if __name__ == "__main__":
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Hyperparameters
    batch_size = 2
    seq_len_q = 5    # Length of query sequence (e.g., text tokens)
    seq_len_kv = 10  # Length of key/value sequence (e.g., image features)
    d_model = 512    # Embedding dimension
    n_heads = 8      # Number of attention heads
    
    # Dummy input tensors, moved to MPS device
    Q = torch.randn(batch_size, seq_len_q, d_model, device=device)
    K = torch.randn(batch_size, seq_len_kv, d_model, device=device)
    V = torch.randn(batch_size, seq_len_kv, d_model, device=device)
    
    # Initialize the cross-attention layer and move to MPS device
    cross_attn = CrossAttentionLayer(d_model=d_model, n_heads=n_heads).to(device)
    
    # Forward pass
    output = cross_attn(Q, K, V)
    print("Output shape:", output.shape)  # Should be [batch_size, seq_len_q, d_model]