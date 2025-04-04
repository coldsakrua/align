import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from data import DataloaderLite
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW
import math
import inspect
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class GPTConfig:
    vocab_size:int=50304    
    n_layer:int=12
    n_head:int=12
    n_embd:int=768
    block_size:int=1024


total_batch_size=2**19
batch_size=16
seq_len=1024
assert total_batch_size%(batch_size*seq_len)==0
grad_accum_steps=total_batch_size//(batch_size*seq_len)
print(f"total_batch_size:{total_batch_size}")
print(f"batch_size:{batch_size}")

class TanhGELU(nn.Module):
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*torch.pow(x,3.0))))     ##避免传输loss

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
        
        # attn=q@k.transpose(-2,-1)*self.scale
        # attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        # attn=attn.softmax(dim=-1)
        # y=attn@v
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)  #flash attention
        
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
        self.transformer.wte.weight=self.lm_head.weight
        self.c_proj.STD_INIT=1
        self.apply(self._init_weights)

    def _init_weights(self,module):
        
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,'INIT_STD'):
                std*=(2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    
    def forward(self,idx, targets=None):
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
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))

            return logits,loss
        else:
            return logits
    
    def configure_optimizers(self,weight_decay,lr,device):
        param_dict={pn:p for pn,p in self.named_parameters() if p.requires_grad}
        decay_params=[p for n,p in param_dict.items() if p.dim()>=2]
        no_decay_params=[p for n,p in param_dict.items() if p.dim()<2]
        optim_groups=[
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':no_decay_params,'weight_decay':0.0}
        ]
        num_decay_params=sum(p.numel() for p in decay_params)
        num_no_decay_params=sum(p.numel() for p in no_decay_params)
        print(f"num of params:{num_decay_params+num_no_decay_params}")
        print(f"num of decay params:{num_decay_params}")
        
        fused_available='fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused=fused_available and device.type=='cuda'
        extra_args=dict(fused=True) if use_fused else dict()
        optimizer=AdamW(optim_groups,lr=lr,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1,**extra_args)
        return optimizer
        
        
          
    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in ['gpt2','gpt2-medium','gpt2-large','gpt2-xl']
        print(f"loading model {model_type}")
        
        config_args={
            'gpt2':dict(n_head=12,n_layer=12,n_embd=768),
        }[model_type]
        config_args['vocab_size']=50304
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



def get_lr(it,warmup_it=10,max_it=50,lr_max=3e-4,lr_min=3e-5):
    if it<warmup_it:
        return lr_max*it/warmup_it
    elif it>max_it:
        return lr_min
    decay_ratio=(it-warmup_it)/(max_it-warmup_it)
    assert 0<=decay_ratio<=1
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))
    return lr_min+coeff*(lr_max-lr_min)
    



def train(model,dataloader,device,epoch):
    model.train()
    model=torch.compile(model)
    optimizer=AdamW(model.parameters(),lr=1e-4,beta=(0.9,0.95),eps=1e-8,weight_decay=0.1)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    for i,(x,y) in enumerate(dataloader):
        loss_num=0
        for micro_step in range(grad_accum_steps):
            #forward
            x,y=dataloader.next_batch()
            x=x.to(device)
            y=y.to(device)
            logits,loss=model(x,y)
            #backward
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                logits,loss=model(x,y)
            loss=loss/grad_accum_steps
            loss_num+=loss.detach()
            loss.backward()
        optimizer.zero_grad()
        
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        lr=get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr']= lr
        optimizer.step()
        scheduler.step()
    
    
if __name__=="__main__":
    # config=GPTConfig()
    model=GPT.from_pretrained('gpt2')
    print("no error!")
    device="cuda" if torch.cuda.is_available() else "cpu"
    max_len=30
    model.to(device)
    model.eval()
    prompt="Hello.Who are you?"
    
    import tiktoken
    from transformers import GPT2Tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('E:/slider and homework/202502/llm-align/hw/hw1-code/gpt')
    tokens=gpt2_tokenizer.encode(prompt)
    tokens=torch.tensor(tokens,dtype=torch.long).unsqueeze(0).repeat(2,1)
    x=tokens.to(device)
    train_loader=DataloaderLite("None",batch=8,seq_len=1024,process_rank=0,num_processes=1,split="train",tokenizer=gpt2_tokenizer)
    
    
    import time
    torch.manual_seed(42)
    start_time=time.time()
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
    end_time=time.time()
    print(f"time cost:{end_time-start_time}")  
    print(f"generated text:{generated_text}")

