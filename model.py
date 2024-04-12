import torch.backends
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm


class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        
        self.attention = nn.Linear(n_embd, 3 * n_embd)
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # batch size, context_size, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.attention(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, context_size, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, context_size, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, context_size, head_size)
        
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0) # (B, n_head, context_size, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super(MLP, self).__init__()
        
        self.n_embd = n_embd
        self.n_hidden = 4 * n_embd
        self.dropout = dropout
        
        self.fc1 = nn.Linear(n_embd, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, n_embd)
    
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super(Block, self).__init__()
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        
        self.attention = SelfAttention(n_embd, n_head, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
        self.layer_norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, n_embd: int, n_head: int, n_layer: int, dropout: float = 0.1):
        super(Transformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(context_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.decoder = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x: torch.Tensor):
        if x.size(1) > self.context_size:
            x = x[:, -self.context_size:]
        
        B, T = x.size()
        
        x = self.token_embedding(x) + self.positional_embedding(torch.arange(T, device=x.device))
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.layer_norm(x)
        x = self.decoder(x)
        
        return x
    
    @torch.no_grad()
    def generate(self, n: torch.Tensor, context: torch.Tensor | None = None, temperature: float = 1.0):
        if context is None:
            context = torch.randint(self.vocab_size, (1,), dtype=torch.long, device=next(self.parameters()).device)
        
        assert context.dim() == 1, "context should be a 1D tensor"
        
        for _ in tqdm(range(n)):
            x = self.forward(context.view(1, -1))
            x = x[0, -1, :] / temperature
            x = F.softmax(x, dim=0)
            context = torch.cat([context, torch.multinomial(x, 1)], dim=0)
        
        return context
    
    def loss(self, x: torch.Tensor, target: torch.Tensor):
        return F.cross_entropy(x.reshape(-1, x.size(-1)), target.reshape(-1))
    
    def accuracy(self, x: torch.Tensor, target: torch.Tensor, topk: int = 1):
        _, pred = x.topk(topk, dim=-1)
        return pred.eq(target.unsqueeze(-1).expand_as(pred)).float().max(-1)[0].mean() # max()[0] to get the max value, max()[1] to get the argmax