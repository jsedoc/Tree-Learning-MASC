# coding: utf-8

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

import random
from queue import *

from seq2tree import *
from tree2seq import *
from seq2seqLoader import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib.pyplot import imshow


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(EncoderLSTM, self).__init__()
        self.word_embeddings = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(input_size = embedding_size, hidden_size = hidden_size,num_layers = 1)
        
        
    def forward(self, input, hidden, c):
        embedded = self.word_embeddings(input).view(1, 1, -1)
        
        output = embedded
        output,(hidden,c) = self.lstm(output, (hidden, c))
        return output,hidden,c

    def initCells(self):
        if use_cuda:
            return Variable(torch.zeros(1, 1, hidden_size).cuda(gpu_no))
        else:
            return Variable(torch.zeros(1, 1, hidden_size))

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda(gpu_no)
        else:
            return result


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initCells()
    encoder_c = encoder.initCells()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
                            
    input_length = len(input_variable)
    target_length = len(target_variable)
    
    loss = 0
    
    if use_cuda:
        encoder_hidden = encoder_hidden.cuda()
        
    for ei in range(input_length):
        encoder_output, encoder_hidden,encoder_c = encoder(input_variable[ei], encoder_hidden,encoder_c)
        
        decoder_hidden = encoder_hidden
        
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                target_variable[di], decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])

        loss.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        return loss.data[0] / target_length
    
        
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = Variable(torch.LongTensor(training_pair[0]).view(-1, 1))
        target_variable = Variable(torch.LongTensor(training_pair[1]).view(-1, 1))
        
        if use_cuda:
            input_variable = input_variable.cuda(gpu_no)
            target_variable = target_variable.cuda(gpu_no)
            
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            torch.save(encoder.state_dict(), './pickles/encoder_seq2seq.pth')
            torch.save(decoder.state_dict(), './pickles/decoder_seq2seq.pth')



train_filename = "data/relabel/train.orig"
target_filename = "data/relabel/train.interior_relabel"
data_loader = S2SDataLoader(train_filename,target_filename)
training_data = data_loader.get_data()
print(len(training_data))
#training_pairs = [(random.choice(training_data))
#                   for i in range(len(training_data))]
training_pairs = training_data
print(training_pairs[0][0])
print(training_pairs[0][1])

hidden_size = 256
embedding_size = 256
gpu_no = 1
# 0-99,+,-,*,/,(,)
encoder_vocab_size = 113
decoder_vocab_size = 117
use_cuda = False
encoder1 = EncoderLSTM(encoder_vocab_size, hidden_size,embedding_size)
decoder1 = DecoderRNN(hidden_size, decoder_vocab_size)


if use_cuda:
    encoder1 = encoder1.cuda(gpu_no)
    decoder1 = decoder1.cuda(gpu_no)

trainIters(encoder1,decoder1, 10000, print_every=100)
