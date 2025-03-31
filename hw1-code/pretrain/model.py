import torch
import torch.nn as nn
from dataclasses import dataclass
@dataclass
class GPTConfig:
    vocab_size:int=50257    
    n_layer=12
    n_head=12
    n_embd=768
    n_ctx=1024
    

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_head=config.n_head
        self.head_dim=config.n_embd//config.n_head
        self.scale=self.head_dim**-0.5
        
        self.register_buffer('mask',torch.tril(torch.ones(config.n_ctx,config.n_ctx)).view(1,1,config.n_ctx,config.n_ctx))
        
        self.q=nn.Linear(config.n_embd,config.n_embd)
        self.k=nn.Linear(config.n_embd,config.n_embd)
        self.v=nn.Linear(config.n_embd,config.n_embd)
        
        self.proj=nn.Linear(config.n_embd,config.n_embd)
    def forward(self,x):
        B,T,C=x.size()
        
        q=self.q(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k=self.k(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v=self.v(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        
        attn=q@k.transpose(-2,-1)*self.scale
        attn=attn.masked_fill(self.mask[:,:,:T,:T]==0,float('-inf'))
        attn=attn.softmax(dim=-1)
        
        y=attn@v
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1=nn.Linear(config.n_embd,4*config.n_embd)
        self.act=nn.GELU(approximate='tanh')
        self.fc2=nn.Linear(4*config.n_embd,config.n_embd)
    
    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.fc2(x)
        return x

class block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
        
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x
        


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd), # token
            wpe=nn.Embedding(config.n_ctx,config.n_embd),  #position
            h=nn.ModuleList([block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)