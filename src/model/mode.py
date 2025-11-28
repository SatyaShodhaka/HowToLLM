# This file implements a transformer model from scratch using PyTorch.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LanguageModel(nn.Module):
    """A simple Transformer-based language model.
    inputs:
        vocab_size: Size of the vocabulary.
        d_model: Dimension of the model (embedding size).
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout rate.
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Defining the embedding layer -> converts token indices(kind of like one-hot vectors) that we get from BPE tokenizer vocab to dense vectors 
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding to add positional information to the embeddings -> implemnts RoPE
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) # Can also use nn.TransformerEncoderLayer from PyTorch
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers) # Can also use nn.TransformerEncoder from PyTorch
        # Final linear layer to project the output to vocabulary size
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the model.
        inputs:
            src: The input sequence (sequence length, batch size).
            src_mask: The mask for the src sequence (optional).
            src_key_padding_mask: The mask for the src keys per batch (optional).
        returns:
            The output logits (sequence length, batch size, vocab size).
        """
        src = self.embedding(src) * math.sqrt(self.d_model)  # Scale embeddings
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask,
                                          src_key_padding_mask=src_key_padding_mask)
        output = self.fc_out(output)
        return output


class TransformerEncoder(nn.Module):
    """A stack of Transformer encoder layers.
    inputs:
        encoder_layer: An instance of TransformerEncoderLayer.
        num_layers: Number of layers to stack.
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)]) #Creates a list of encoder layers
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """Pass the input through the stacked encoder layers.
        inputs:
            src: The input sequence (sequence length, batch size, d_model).
            mask: The mask for the src sequence (optional).
            src_key_padding_mask: The mask for the src keys per batch (optional).
        returns:
            The output of the encoder (sequence length, batch size, d_model).
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)
        return output

class TransformerEncoderLayer(nn.Module):
    """A single layer of the Transformer encoder.
    inputs:
        d_model: Dimension of the model (embedding size).
        nhead: Number of attention heads.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout rate.

    resource:
    Created by me - https://excalidraw.com/#json=JDqI68RlAXP9pXB6H98nD,9NTtBrPBklcy-FRns_ik4Q
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) #Using the built-in multi-head attention module from PyTorch but can also be implemented from scratch
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        """ 
        Pass the input through the encoder layer.
        inputs:
            src: The input sequence (sequence length, batch size, d_model).
            src_mask: The mask for the src sequence (optional).
            src_key_padding_mask: The mask for the src keys per batch (optional). This is used to mask out the padding tokens in the input sequence.
        returns:
            The output of the encoder layer (sequence length, batch size, d_model).
        input flow:
         src (sequence length, batch size, d_model) -> Self-Attention (Computes q, k, v; divides them based on number of heads and then attention scores for each head followed by concatenation and linear transformation to get back the original dimension) -> Add & Norm -> Feedforward -> Add & Norm -> output
        """
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask) # Returns a tuple (attn_output, attn_output_weights)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
        


class PositionalEncoding(nn.Module):    
    """Implements Rotary Positional Encoding (RoPE) for transformer models.
    inputs:
        d_model: Dimension of the model (embedding size).
        dropout: Dropout rate.
        max_len: Maximum length of input sequences.
    retuns: A tensor with positional encodings added to the input embeddings.
    resources: https://medium.com/ai-insights-cobet/rotary-positional-embeddings-a-detailed-look-and-comprehensive-understanding-4ff66a874d83
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)