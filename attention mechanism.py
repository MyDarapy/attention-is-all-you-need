import torch 
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, head_size, embed_dim, block_size, dropout_prob=0.1, masked_attention=True):
        super(Attention, self).__init__()
        self.head_size = head_size 
        self.masked = masked_attention 
        
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        if self.masked:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, head_size):
        batch, block_size, embedding_dim = input.shape
        q = self.query(input) -> (B,T,C)
        k = self.key(input) -> (B,T,C)
        v = self.value(input) -> (B,T,C)

        attention_scores = q@k.transpose(-2,-1) -> (B,T,C) @ (B, C, T) @ (B, T, T)
        scaled_scores = attention_scores / self.head_size**0.5
        if masked: 
            scaled_scores = scaled_scores.masked_fill(self.tril[:block_size, :block_size]==0, float('-inf'))
        attention weights = self.softmax(scaled_scores)
        attention_weights = self.dropout(attention_weights)

        attention_vectors = attention_weights@v
        return attention_vectors

class MultiAttentionHeads(nn.Module):
    def __init__(self, head_size, num_of_heads, dropout_prob=0.1, masked_attention=True):
        super(MultiAttentionHeads, self).__init__()
        self.head_size = head_size
        self.num_of_heads = num_of_heads
        self.attention_heads =  nn.ModuleList([
            Attention(head_size, embed_dim, block_size, dropout_prob, masked_attention) for _ in range(num_of_heads)])

        self.projection = nn.Linear(head_size*num_of_heads, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        concatenated_heads = torch.cat([h(x) for h in self.attention_heads], dim=-1)
        attention_vectors = self.projection(oncatenated_heads)
        attention_vectors = self.dropout(attention_vectors)
        return attention_vectors














