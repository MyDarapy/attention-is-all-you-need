class FeedforwardLayer(nn.Module):
    def __init__(self, embed_dim, dropout_prob=0.1):
        super(FeedforwardLayer, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim), nn.ReLU(), 
            nn.Linear(embed_dim*4, embed_dim), nn.dropout(dropout_prob))
    
    def forward(self, x):
        return self.ff(x)



