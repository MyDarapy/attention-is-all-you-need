import torch 
import torch.nn
from src.attention_mechanism import MultiHeadAttention
from src.utils import EmbeddingLayer
from src.Feedforward import FeedforwardLayer


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, block_size, num_of_heads, dropout_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.num_of_heads = num_of_heads
        assert embed_dim // num_of_heads == 0, f'model embedding dimension:{embed_dim} is not divisible by number of heads:{num_of_heads}'
        self.head_size = embed_dim // num_of_heads

        self.masked_attention_head = MultiHeadAttention(
            self.head_size, self.num_of_heads, dropout_prob, masked=True) 

        self.multi_attention_heads= MultiHeadAttention(
            self.head_size, self.num_of_heads, dropout_prob, masked=False) 

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)

        self.feedforward = FeedforwardLayer(embed_dim)

    def forward(self, encoder_output, decoder_input):
        masked_attention_vectors = self.dropout(self.masked_attention_heads(decoder_input, decoder_input, decoder_input))
        masked_attention_vectors = self.layernorm1(decoder_input + masked_attention_vectors)

        multi_attention_head_vectors = self.multi_attention_heads(
            encoder_output, decoder_input, decoder_input)
        multi_attention_head_vectors = self.dropout(multi_attention_head_vectors)
        multi_attention_head_vectors = self.laynorm2(masked_attention_vectors + multi_attention_head_vectors)
        attention_vector = self.feedforward(multi_attention_head_vectors)
        ff_output = self.dropout(attention_vector)
        attention_vector = self.layernorm3(ff_output + multi_attention_head_vectors)

        return attention_vectors 

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_of_layers, block_size, num_of_heads, dropout_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.num_of_layers = num_of_layers
        self.decoder = nn.ModuleList([DecoderLayer(
            embed_dim, block_size, num_of_heads, dropout_prob=0.1) for _ in range(num_of_layers)])
        self.language_modeling_head = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, encoder_output, trgt_embeddings):
        output = trgt_embeddings
        for decoder_layer in self.decoder:
            output = decoder_layer(encoder_output, output)
        output= self.language_modelling_head(output)
        logits = self.softmax(output)
        return logits 





        