import torch
from torch.utils.data import Dataset,DataLoader

class DataloaderLite:
    def __init__(self,batch,seq_len,process_rank,num_processes,split):
        self.batch=batch
        self.seq_len=seq_len
        self.process_rank=process_rank
        self.num_processes=num_processes
        assert split in ['train','val']
        self.split=split
        
        


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
        
        
