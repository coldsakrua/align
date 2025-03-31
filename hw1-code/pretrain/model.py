import torch
import torch.nn as nn
from dataclasses import dataclass
@dataclass
class GPTConfig:
    vocab_size:int=50257    
    n_layer:int=12
    n_head:int=12
    n_embd:int=768
    block_size:int=1024
    

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_head=config.n_head
        self.head_dim=config.n_embd//config.n_head
        self.scale=self.head_dim**-0.5
        
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
        
        self.qkv=nn.Linear(config.n_embd,config.n_embd*3)

        
        self.proj=nn.Linear(config.n_embd,config.n_embd)
    def forward(self,x):
        B,T,C=x.size()
        q,k,v=self.qkv(x).split(C,dim=-1)
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
            wpe=nn.Embedding(config.block_size,config.n_embd),  #position
            h=nn.ModuleList([block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        
        
    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in ['gpt2','gpt2-medium','gpt2-large','gpt2-xl']
        print(f"loading model {model_type}")
        
        config_args={
            'gpt2':dict(n_head=12,n_layer=12,n_embd=768),
        }[model_type]
        config_args['vocab_size']=50257
        config_args['block_size']=1024
        print(config_args)
        config=GPTConfig(**config_args)
        model=GPT(config)
        
        sd=model.state_dict()
        sd_keys=sd.keys()
        sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')]
        from transformers import GPT2LMHeadModel
        model_hf=GPT2LMHeadModel.from_pretrained('E:/slider and homework/202502/llm-align/hw/hw1-code/gpt')
        
        sd_hf=model_hf.state_dict()
        sd_hf_keys=sd_hf.keys()
        sd_hf_keys=[k for k in sd_hf_keys if not k.endswith('.attn.masked_bias')]
        sd_hf_keys=[k for k in sd_hf_keys if not k.endswith('.attn.bias')]
        transposed=['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
        assert len(sd_keys)==len(sd_hf_keys)
        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k]=sd_hf[k].T
            else:
                with torch.no_grad():
                    sd[k]=sd_hf[k]
        return model
if __name__=="__main__":
    # config=GPTConfig()
    model=GPT.from_pretrained('gpt2')
    print("no error!")
    