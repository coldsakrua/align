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
        
        self.c_attn=nn.Linear(config.n_embd,config.n_embd*3)

        
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
    def forward(self,x):
        B,T,C=x.size()
        q,k,v=self.c_attn(x).split(C,dim=-1)
        q=q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        
        attn=q@k.transpose(-2,-1)*self.scale
        attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        attn=attn.softmax(dim=-1)
        
        y=attn@v
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.act=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.act(x)
        x=self.c_proj(x)
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
        
    
    def forward(self,idx):
        B,T=idx.size()
        assert T<=self.config.block_size
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_ebd=self.transformer.wpe(pos)
        tok_ebd=self.transformer.wte(idx)
        x=pos_ebd+tok_ebd
        
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        
        logits=self.lm_head(x)
        return logits
    
    
    
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
            if any(k.endswith(ww) for ww in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
if __name__=="__main__":
    # config=GPTConfig()
    model=GPT.from_pretrained('gpt2')
    print("no error!")
    
    max_len=30
    model.to("cuda")
    model.eval()
    prompt="Hello.Who are you?"
    
    import tiktoken
    from transformers import GPT2Tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('E:/slider and homework/202502/llm-align/hw/hw1-code/gpt')
    tokens=gpt2_tokenizer.encode(prompt)
    tokens=torch.tensor(tokens,dtype=torch.long).unsqueeze(0).repeat(2,1)
    x=tokens.to("cuda")
    
    torch.manual_seed(42)
    while(x.size(1)<max_len):
        with torch.no_grad():
            y = model(x)
            y = y[:, x.size(1)-1, :]
            # 使用top-k采样
            k = 40
            v, _ = torch.topk(y, k)
            prob = torch.softmax(v, dim=-1)
            choice = torch.multinomial(prob, num_samples=1)
            next_token = _.gather(-1, choice)
            x = torch.cat([x, next_token], dim=1)
            generated_text = gpt2_tokenizer.decode(x[0].tolist())
            if next_token[0] == gpt2_tokenizer.eos_token_id:
                break
            
    print(f"generated:{generated_text}")