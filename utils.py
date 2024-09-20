import torch 
import torch.nn

#Implement Sinusoidial position encoding
def get_position_encoding():
    pass 

class EmbeddingLayer(nn.Module):
    def __init__ (self, vocab_size, embed_dim, block_size):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        self.positional_encoding = nn.Embedding(block_size, embed_dim) #Learned Embeddings


    def forward(self, input_ids):
        batch, block_size = input_ids.shape -> (B,T)
        token_embeddings = self.embeddings(input_ids) -> (N, embed_dim)
        position_embeddings = self.positional_embeddings(torch.arange(block_size, device: torch.device = torch.device("cpu")))
        token = token_embeddings + position_embeddings
        return token


class SkipConnections():
    pass
    

