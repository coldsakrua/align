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
from helloswag import render_example

@dataclass
class GPTConfig:
    vocab_size:int=50304    
    n_layer:int=12
    n_head:int=12
    n_embd:int=768
    block_size:int=1024


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
    

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# def train(model,dataloader,device,epoch,ddp,grad_accum_steps):
    
    
if __name__=="__main__": 
    use_compile=False   
    from torch.distributed import init_process_group, destroy_process_group

    ddp=int(os.environ.get("RANK",1))!=-1

    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend="nccl")
        ddp_rank=int(os.environ["RANK"])
        ddp_local_rank=int(os.environ["LOCAL_RANK"])
        ddp_world_size=int(os.environ["WORLD_SIZE"])
        device=torch.device(f"cuda:{ddp_local_rank}")
        device=f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process=ddp_rank==0

    else:
        ddp_rank=0
        ddp_local_rank=0
        ddp_world_size=1
        device="cuda" if torch.cuda.is_available() else "cpu"
        master_process=True


    total_batch_size=2**19
    batch_size=16
    seq_len=1024
    assert total_batch_size%(batch_size*seq_len*ddp_world_size)==0
    grad_accum_steps=total_batch_size//(batch_size*seq_len*ddp_world_size)
    print(f"total_batch_size:{total_batch_size}")
    print(f"batch_size:{batch_size}")
    # config=GPTConfig()
    
    model=GPT.from_pretrained('gpt2')
    print("no error!")
    max_len=30
    if use_compile:
        model=torch.compile(model)
    epoch=1000
    dataloader=DataloaderLite("None",batch=batch_size,seq_len=seq_len,process_rank=ddp_rank,num_processes=ddp_world_size,split="train")
    val_loader=DataloaderLite("None",batch=8,seq_len=1024,process_rank=0,num_processes=1,split="val",tokenizer=gpt2_tokenizer)
    if ddp:
        model=DDP(model,device_ids=[ddp_local_rank])
    raw_model=model.module if ddp else model
    val_loss_step=20
    # train(raw_model,train_loader,device,1,ddp,grad_accum_steps=grad_accum_steps)
    model.train()
    optimizer=raw_model.configure_optimizers(weight_decay=0.1,lr=1e-4,device=device)
    # optimizer=AdamW(model.parameters(),lr=1e-4,beta=(0.9,0.95),eps=1e-8,weight_decay=0.1)
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    log_dir="log"
    os.makedirs(log_dir,exist_ok=True)
    log_file=os.path.join(log_dir,"log.txt")
    with open(log_file,"w") as f:
        # f.write("")
        pass
    
    for i in tqdm(range(epoch)):
        last_step=(i==epoch-1)
        if i%100==0:
            model.eval()
            val_loader.reset()
            
            with torch.no_grad():
                val_loss=0
                val_num=0
                for _ in range(val_loss_step):
                    x,y=val_loader.next_batch()
                    x=x.to(device)
                    y=y.to(device)
                    with torch.autocast(device_type=device,dtype=torch.bfloat16):
                        logits,loss=model(x,y)
                    val_loss+=loss.detach()/val_loss_step
            if ddp:
                dist.all_reduce(val_loss,op=dist.ReduceOp.AVG)
            if master_process:
                print(f"epoch:{i},val_loss:{val_loss}")
        
        
        if (i%250==0 or last_step)  and(not use_compile):
            num_correct_num=0
            num_total=0
            for i,example in enumerate(iterate_examples("val")):
                if i%ddp_world_size==ddp_rank:
                    data, tokens, mask, label = render_example(example)
                    tokens = tokens.to(device)
                    mask = mask.to(device)
                    with torch.no_grad():
                        with torch.autocast(device_type=device,dtype=torch.bfloat16):
                            logits,loss=model(tokens)
                        pred_norm=get_most_likely_row(tokens, mask, logits)
                        num_correct_num+=int(pred_norm==label)
                        num_total+=1
                else:
                    continue
            if ddp:
                num_total=torch.tensor(num_total,dtype=torch.long,device=device)
                num_correct_num=torch.tensor(num_correct_num,dtype=torch.long,device=device)
                dist.all_reduce(torch.tensor(num_correct_num),op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(num_total),op=dist.ReduceOp.SUM)
                num_total=num_total.item()
                num_correct_num=num_correct_num.item()
            acc_norm=num_correct_num/num_total
            if master_process:
                print(f"epoch:{i},val_acc:{num_correct_num/num_total}")
                with open(log_file,"a") as f:
                    f.write(f"epoch:{i},val_acc:{num_correct_num/num_total}\n")
        
        loss_num=0
        for micro_step in range(grad_accum_steps):
            #forward
            x,y=dataloader.next_batch()
            x=x.to(device)
            y=y.to(device)
            logits,loss=model(x,y)
            #backward
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits,loss=model(x,y)
            loss=loss/grad_accum_steps
            loss_num+=loss.detach()
            if ddp:
                model.require_backward_grad_sync=(micro_step==accumulation_steps-1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_num,op=dist.ReduceOp.AVG)
        optimizer.zero_grad()
        
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        lr=get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr']= lr
        optimizer.step()
        # scheduler.step()
    if ddp:
        destroy_process_group()
        
    
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
    num_return_general=4
    while(x.size(1)<max_len):
        with torch.no_grad():
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                y,loss=model(x)

            y = y[:, -1, :]
            y = torch.softmax(v, dim=-1)
            # 使用top-k采样
            k = 40
            v, _ = torch.topk(y, k)
            choice = torch.multinomial(prob, num_samples=1)
            next_token = _.gather(-1, choice)
            x = torch.cat([x, next_token], dim=1)
            
    for i in range(num_return_general):
        tokens=x[i:max_len].tolist()
        generated_text = gpt2_tokenizer.decode(tokens)
        print(f"rank{ddp_rank} sample{i}:{generated_text}")
    end_time=time.time()
    print(f"time cost:{end_time-start_time}")  
    print(f"generated text:{generated_text}")
   
    
    