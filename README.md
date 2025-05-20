# Simple GPT-2 Implementation

This is a simplified implementation of a GPT-2 style transformer layer in PyTorch. It demonstrates the core components of a transformer architecture including multi-head attention and feed-forward networks.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the example:

```bash
python simple_gpt2.py
```

## Components

The implementation includes:

1. `MultiHeadAttention`: Implements the multi-head self-attention mechanism

   - Scaled dot-product attention
   - Multiple attention heads
   - Projection layers

2. `TransformerBlock`: A complete transformer block containing:
   - Multi-head attention
   - Layer normalization
   - Feed-forward network
   - Residual connections

## Example

The main script demonstrates how to:

- Create a transformer block
- Process input data through the transformer
- View the input and output shapes

The example uses:

- Batch size: 2
- Sequence length: 10
- Model dimension: 256
- Number of attention heads: 4
- Feed-forward dimension: 1024
