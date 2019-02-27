import torch.nn as nn
from word_sequence import num_sequence
import config


class NumEncoder(nn.Module):
    def __init__(self):
        super(NumEncoder,self).__init__()
        self.vocab_size = len(num_sequence)
        self.dropout = config.dropout
        self.embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_dim,padding_idx=num_sequence.PAD)
        self.gru = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=1,
                          batch_first=True,
                          dropout=config.dropout)

    def forward(self, input,input_length):
        embeded = self.embedding(input)
        # embeded = embeded.permute(1,0,2) #[seq_len, batch_size, embedding_dim]
        # print("embed size", embeded.size())
        embeded = nn.utils.rnn.pack_padded_sequence(embeded,lengths=input_length,batch_first=True)

        #hidden:[1,batch_size,vocab_size]
        out,hidden = self.gru(embeded)
        out,outputs_length = nn.utils.rnn.pad_packed_sequence(out,batch_first=True,padding_value=num_sequence.PAD)
        #hidden #[batch_size,hidden_size]
        # out [batch_size,seq_len,hidden_size]
        # hidden = hidden.squeeze(0)
        #hidden [1,batch_size,hidden_size]
        return out,hidden