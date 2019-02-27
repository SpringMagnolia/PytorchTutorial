import torch
import torch.nn as nn
import config
import random
import torch.nn.functional as F
from word_sequence import num_sequence

class NumDecoder(nn.Module):
    def __init__(self):
        super(NumDecoder,self).__init__()
        self.max_seq_len = config.max_len
        self.vocab_size = len(num_sequence)
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_dim,padding_idx=num_sequence.PAD)
        self.gru = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=1,
                          batch_first=True,
                          dropout=self.dropout)
        self.log_softmax = nn.LogSoftmax()

        self.fc = nn.Linear(config.hidden_size,self.vocab_size)

    def forward(self, encoder_hidden,target,target_length):
        # encoder_hidden [batch_size,hidden_size]
        # target [batch_size,seq-len]

        decoder_input = torch.LongTensor([[num_sequence.SOS]]*config.batch_size)
        # print("decoder_input size:",decoder_input.size())
        decoder_outputs = torch.zeros(config.batch_size,config.max_len,self.vocab_size) #[seq_len,batch_size,14]

        decoder_hidden = encoder_hidden #[batch_size,hidden_size]

        for t in range(config.max_len):
            decoder_output_t , decoder_hidden = self.forward_step(decoder_input,decoder_hidden)
            # print(decoder_output_t.size(),decoder_hidden.size())
            # print(decoder_outputs.size())
            decoder_outputs[:,t,:] = decoder_output_t

            use_teacher_forcing = random.random() > 0.5
            if use_teacher_forcing:
                decoder_input =target[:,t].unsqueeze(1)  #[batch_size,1]
            else:
                value, index = torch.topk(decoder_output_t, 1) # index [batch_size,1]
                decoder_input = index
            # print("decoder_input size:",decoder_input.size(),use_teacher_forcing)
        return decoder_outputs,decoder_hidden

    def forward_step(self,decoder_input,decoder_hidden):
        """
        :param decoder_input:[batch_size,1]
        :param decoder_hidden: [1,batch_size,hidden_size]
        :return: out:[batch_size,vocab_size],decoder_hidden:[1,batch_size,didden_size]
        """
        embeded = self.embedding(decoder_input)  #embeded: [batch_size,1 , embedding_dim]
        # print("forworad step embeded:",embeded.size())
        out,decoder_hidden = self.gru(embeded,decoder_hidden) #out [1, batch_size, hidden_size]
        # print("forward_step out size:",out.size()) #[1, batch_size, hidden_size]
        out = out.squeeze(0)
        out = F.log_softmax(self.fc(out),dim=-1)#[batch_Size, vocab_size]
        out = out.squeeze(1)
        # print("out size:",out.size(),decoder_hidden.size())
        return out,decoder_hidden

    def evaluation(self,encoder_hidden): #[1, 20, 14]
        # target = target.transpose(0, 1)  # batch_first = False
        batch_size = encoder_hidden.size(1)

        decoder_input = torch.LongTensor([[num_sequence.SOS] * batch_size])
        # print("decoder start input size:",decoder_input.size()) #[1, 20]
        decoder_outputs = torch.zeros(batch_size,config.max_len, self.vocab_size)  # [seq_len,batch_size,14]
        decoder_hidden = encoder_hidden

        for t in range(config.max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:,t,:] = decoder_output_t
            value, index = torch.topk(decoder_output_t, 1)  # index [20,1]
            decoder_input = index.transpose(0, 1)

        # print("decoder_outputs size:",decoder_outputs.size())
        # # 获取输出的id
        decoder_indices =[]
        # decoder_outputs = decoder_outputs.transpose(0,1) #[batch_size,seq_len,vocab_size]
        # print("decoder_outputs size",decoder_outputs.size())
        for i in range(decoder_outputs.size(1)):
            value,indices = torch.topk(decoder_outputs[:,i,:],1)
            # print("indices size",indices.size(),indices)
            # indices  = indices.transpose(0,1)
            decoder_indices.append(int(indices[0][0].data))
        return decoder_indices

