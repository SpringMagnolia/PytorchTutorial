import torch
import config
from torch import optim
import torch.nn as nn
from encoder import NumEncoder
from decoder import NumDecoder
from seq2seq import Seq2Seq
from dataset import data_loader as train_dataloader
from word_sequence import num_sequence



encoder = NumEncoder()
decoder = NumDecoder()
model = Seq2Seq(encoder,decoder)
print(model)
print("\nInitializing weights...")
for name, param in model.named_parameters():
    if 'bias' in name:
        torch.nn.init.constant_(param, 0.0)
    elif 'weight' in name:
        torch.nn.init.xavier_normal_(param)

# model.load_state_dict(torch.load("model/seq2seq_model.pkl"))
optimizer =  optim.Adam(model.parameters())
# optimizer.load_state_dict(torch.load("model/seq2seq_optimizer.pkl"))
criterion= nn.NLLLoss(ignore_index=num_sequence.PAD,reduction="mean")

def get_loss(decoder_outputs,target):
    target = target.view(-1) #[batch_size*max_len]
    decoder_outputs = decoder_outputs.view(config.batch_size*config.max_len,-1)
    return criterion(decoder_outputs,target)


def train(epoch):
    for idx,(input,target,input_length,target_len) in enumerate(train_dataloader):
        optimizer.zero_grad()
        ##[seq_len,batch_size,vocab_size] [batch_size,seq_len]
        decoder_outputs,decoder_hidden = model(input,target,input_length,target_len)
        loss = get_loss(decoder_outputs,target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, idx * len(input), len(train_dataloader.dataset),
                   100. * idx / len(train_dataloader), loss.item()))

        # torch.save(model.state_dict(), "model/seq2seq_model.pkl")
        # torch.save(optimizer.state_dict(), 'model/seq2seq_optimizer.pkl')

if __name__ == '__main__':
    for i in range(10):
        train(i)