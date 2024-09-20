import torch 
import torch.nn as nn 
from src.decoder import Decoder
from src.decoder import Encoder
from src.utils import EmbeddingLayer


class Transformer(nn.Module):
    def __init__(self, ):
        super(Transformer, self).__init__(ocab_size, embed_dim, num_of_layers, block_size, num_of_heads, dropout_prob=0.1)
        self.embeddings = EmbeddingLayer(vocab_size, embed_dim, block_size)
        self.encoder = EncoderLayer(vocab_size, embed_dim, num_of_layers, block_size, num_of_heads, dropout_prob=0.1)
        self.decoder = DecoderLayer(vocab_size, embed_dim, num_of_layers, block_size, num_of_heads, dropout_prob=0.1)
        
    def forward(self, source, target):
        source_embeddings = source 
        target_embeddings = target 
        encoder_output = self.encoder(source_embeddings)
        decoder_output = self.decoder(encoder_output, target_embeddings)
        return decoder_output
