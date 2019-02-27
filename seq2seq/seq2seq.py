import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input,target,input_length,target_length):
        encoder_outputs,encoder_hidden = self.encoder(input,input_length)
        decoder_outputs,decoder_hidden = self.decoder(encoder_hidden,target,target_length)
        return decoder_outputs,decoder_hidden

    def evaluation(self,inputs,input_length):
        encoder_outputs,encoder_hidden = self.encoder(inputs,input_length)
        decoded_sentence = self.decoder.evaluation(encoder_hidden)
        return decoded_sentence