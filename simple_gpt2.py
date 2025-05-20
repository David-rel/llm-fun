import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = embedding_dim // num_attention_heads
        
        self.query_key_value = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Create Query, Key, Value matrices
        query_key_value = self.query_key_value(hidden_states)
        query_key_value = query_key_value.reshape(batch_size, sequence_length, 3, self.num_attention_heads, self.attention_head_size)
        query_key_value = query_key_value.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value)
        
        # Reshape and project
        context_layer = context_layer.transpose(1, 2).reshape(batch_size, sequence_length, self.embedding_dim)
        return self.output_projection(context_layer)

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, intermediate_size):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_attention_heads)
        self.attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(embedding_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, embedding_dim)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output = self.attention(self.attention_layer_norm(hidden_states), attention_mask)
        hidden_states = hidden_states + attention_output
        
        # Feed-forward
        feed_forward_output = self.feed_forward(self.feed_forward_layer_norm(hidden_states))
        hidden_states = hidden_states + feed_forward_output
        
        return hidden_states

def main():
    # Example usage
    batch_size = 2
    sequence_length = 10
    embedding_dim = 256
    num_attention_heads = 4
    intermediate_size = 1024
    
    # Create a sample input
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)
    
    # Create transformer block
    transformer = TransformerBlock(embedding_dim, num_attention_heads, intermediate_size)
    
    # Forward pass
    output_tensor = transformer(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

if __name__ == "__main__":
    main() 