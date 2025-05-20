import torch
import torch.nn as nn
import torch.nn.functional as F
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

class SimpleGPT2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_attention_heads, intermediate_size, num_layers):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(100, embedding_dim) # Max 100 positions
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_attention_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        sequence_length = input_ids.size(1)
        
        # Get token embeddings
        hidden_states = self.token_embeddings(input_ids)
        
        # Add position embeddings
        position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)
            
        # Final layer norm and projection
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        return logits

def create_simple_tokenizer(text):
    """Create a very simple character-level tokenizer"""
    chars = sorted(list(set(text)))
    vocab = {char: i for i, char in enumerate(chars)}
    vocab['<pad>'] = len(vocab)
    
    # Add inverse mapping
    id_to_token = {i: char for char, i in vocab.items()}
    
    return vocab, id_to_token

def main():
    # Our simple training text
    text = "the cat sat on the mat"
    
    # Create tokenizer and vocabulary
    vocab, id_to_token = create_simple_tokenizer(text)
    vocab_size = len(vocab)
    
    # Tokenize input
    input_ids = torch.tensor([[vocab[char] for char in text]], dtype=torch.long)
    
    # Model parameters
    embedding_dim = 64
    num_attention_heads = 2
    intermediate_size = 128
    num_layers = 2
    
    # Create model
    model = SimpleGPT2(vocab_size, embedding_dim, num_attention_heads, intermediate_size, num_layers)
    
    # Training parameters
    num_epochs = 1000
    learning_rate = 0.001
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        logits = model(input_ids)
        
        # Prepare targets (shifted right)
        targets = torch.zeros_like(input_ids)
        targets[0, :-1] = input_ids[0, 1:]
        targets[0, -1] = input_ids[0, 0]  # Wrap around for simplicity
        
        # Compute loss
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Generate text after training
    print("\nGenerating text:")
    
    # Start with the first character
    input_char = "t"
    input_id = torch.tensor([[vocab[input_char]]], dtype=torch.long)
    generated_text = input_char
    
    # Generate 20 more characters
    with torch.no_grad():
        for _ in range(20):
            # Get predictions
            logits = model(input_id)
            next_token_logits = logits[0, -1, :]
            
            # Get the most likely next token
            next_token_id = torch.argmax(next_token_logits).item()
            next_char = id_to_token[next_token_id]
            
            # Add to generated text
            generated_text += next_char
            
            # Update input for next iteration
            input_id = torch.cat([input_id, torch.tensor([[next_token_id]])], dim=1)
    
    print(f"Input: {input_char}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main() 