from src.attention_mechanism import MultiHeadAttention
from src.utils import EmbeddingLayer
from src.Feedforward import FeedforwardLayer
import torch 
import torch.nn

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, block_size, num_of_heads, dropout_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.num_of_heads = num_of_heads
        assert embed_dim // num_of_heads == 0, f'model embedding dimension:{embed_dim} is not divisible by number of heads:{num_of_heads}'
        self.head_size = embed_dim // num_of_heads
        self.multi_attention_heads= MultiHeadAttention(
            self.head_size, self.num_of_heads, dropout_prob, masked=False) '''Masked is set to false in the encoder 
            layer because the positions are all allowed to talk to each other and see into the future'''' 


        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.feedforward = FeedforwardLayer(embed_dim)

    def forward(self, encoder_input):
        attention_vectors = self.layernorm1(encoder_input + self.multi_attention_heads(
            encoder_input, encoder_input, encoder_input)) #Implement skip connections
        feedforward_output = self.feedforward(attention_vectors)
        attention_vectors = self.layernorm2(attention_vectors + feedforward_output)
        return attention_vectors 


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_of_layers, block_size, num_of_heads, dropout_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.num_of_layers = num_of_layers
        self.encoder = nn.ModuleList([EncoderLayer(
            embed_dim, block_size, num_of_heads, dropout_prob=0.1) for _ in range(num_of_layers)])
    
    def forward(self, src_embeddings):
        output = src_embeddings
        for encoder_layer in self.encoder:
            output = encoder_layer(output)
        return output
    
    
        


    


    








