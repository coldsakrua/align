import torch
from torch.utils.data import Dataset,DataLoader
from transformers import GPT2Tokenizer


class DataloaderLite:
    def __init__(self,data_root,batch,seq_len,process_rank,num_processes,split,tokenizer):
        self.data_root=data_root
        self.batch=batch
        self.seq_len=seq_len
        self.process_rank=process_rank
        self.num_processes=num_processes
        self.current_batch=0
        assert split in ['train','val']
        self.split=split
        
        with open(os.path.join(self.data_root)) as f:
            self.data=f.read()

        self.tokenizer=tokenizer
        self.tokens=torch.tensor(self.tokenizer.encode(self.data),dtype=torch.long)

        print(f"total tokens:{len(self.tokens)}")
        print(f"batches:{len(self.tokens)//(self.batch*self.seq_len)}")

    def next_batch(self):
        start_idx=self.current_batch*self.batch*self.seq_len
        end_idx=(self.current_batch+1)*self.batch*self.seq_len
        buf=self.tokens[start_idx:end_idx]
        x=buf[:self.seq_len].view(self.batch,self.seq_len)
        y=buf[1:self.seq_len+1].view(self.batch,self.seq_len)
        if buf.size(0)<self.seq_len+1:
            x=torch.cat([x,torch.zeros(self.batch,self.seq_len-buf.size(0)+1,dtype=torch.long)],dim=1)
            y=torch.cat([y,torch.zeros(self.batch,self.seq_len-buf.size(0)+1,dtype=torch.long)],dim=1)
        self.current_batch+=1
        return x,y
    
    def reset_batch_pointer(self):
        self.current_batch=0
        
