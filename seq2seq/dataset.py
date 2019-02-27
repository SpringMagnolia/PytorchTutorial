from torch.utils.data import Dataset,DataLoader
import numpy as np
from word_sequence import num_sequence
import torch
import config

class RandomDataset(Dataset):
    def __init__(self):
        super(RandomDataset,self).__init__()
        self.total_data_size = 500000
        np.random.seed(10)
        self.total_data = np.random.randint(1,100000000,size=[self.total_data_size])

    def __getitem__(self, idx):
        input = str(self.total_data[idx])
        return input, input+ "0",len(input),len(input)+1

    def __len__(self):
        return self.total_data_size

def collate_fn(batch):
    #1. 对batch进行排序，按照长度从长到短的顺序排序
    batch = sorted(batch,key=lambda x:x[3],reverse=True)
    input,target,input_length,target_length = zip(*batch)

    #2.进行padding的操作
    input = torch.LongTensor([num_sequence.transform(i,max_len=config.max_len) for i in input])
    target = torch.LongTensor([num_sequence.transform(i,max_len=config.max_len,add_eos=True) for i in target])
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)

    return input,target,input_length,target_length

data_loader = DataLoader(dataset=RandomDataset(),batch_size=config.batch_size,collate_fn=collate_fn,drop_last=True)

if __name__ == '__main__':
    data_loader = DataLoader(dataset=RandomDataset(),batch_size=config.batch_size,drop_last=True)
    for idx,(input,target,input_lenght,target_length) in enumerate(data_loader):
        print(idx)
        print(input)
        print(target)
        print(input_lenght)
        print(target_length)
        break
