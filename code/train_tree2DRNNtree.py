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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib.pyplot import imshow

def generate_training_data(n_iters,train_filename,target_filename):
    data_loader = Tree2TreeDataLoader(train_filename,target_filename)
    training_data = data_loader.get_data()
    print(len(training_data))
    training_pairs = [(random.choice(training_data))
                      for i in range(n_iters)]
    print(training_pairs[0])
    print(training_pairs[0][1].key)
    return training_pairs


# In[6]:


plot_loss_avg = []
plot_losses = []


# In[7]:


class Tree:
    def __init__(self,level,rootObj=None):
        self.level = level
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None
        self.parent = None
        self.sibling = None
        self.children = [] 
        self.hiddenA = None
        self.hiddenF = None

class DecoderDRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, decoder_vocab_size):
        super(DecoderDRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(decoder_vocab_size, embedding_dim)
        self.gruA = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim,num_layers = 1)
        self.gruF = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim,num_layers = 1)
        self.linearA1 = nn.Linear(hidden_dim,hidden_dim)
        self.linearF1 = nn.Linear(hidden_dim,hidden_dim)
        self.projA = nn.Linear(hidden_dim,1)
        self.projF = nn.Linear(hidden_dim,1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.vA = nn.Linear(1,decoder_vocab_size)
        self.vF = nn.Linear(1,decoder_vocab_size)
        self.predWeight = nn.Linear(hidden_dim,decoder_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        init_bias = torch.from_numpy(-np.ones((1,3*hidden_dim)))
        self.gruA.bias_hh_l0.data = init_bias.float()
        self.gruF.bias_hh_l0.data = init_bias.float()
    
    def forward(self, hiddenA, hiddenF, parent, sibling):
        embeddedA = self.word_embeddings(parent).view(1, 1, -1)
        embeddedF = self.word_embeddings(sibling).view(1, 1, -1)
        outputA = embeddedA
        outputF = embeddedF
        outputA, hiddenA = self.gruA(outputA,hiddenA)
        outputF, hiddenF = self.gruF(outputF,hiddenF)
        hPred = self.tanh(self.linearA1(hiddenA.view(1,-1))+self.linearF1(hiddenF.view(1,-1)))
        childPred = self.sigmoid(self.projA(hPred))
        siblingPred = self.sigmoid(self.projF(hPred))
        
            
        output = self.predWeight(hPred.view(1,-1))
        if(torch.gt(childPred,0.5).data[0][0]):
            output += self.vA(childPred)
        if(torch.gt(siblingPred,0.5).data[0][0]):
            output += self.vF(siblingPred)       
        
        output = self.log_softmax(output)
        return hiddenA,hiddenF,childPred,siblingPred,output

def train(input_variable, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion_label,criterion_topology):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()


    target_length = input_variable[1].size()[0]
    tree_matrix = Variable(input_variable[1])
    tree_matrix = tree_matrix.cuda(gpu_no) if use_cuda else tree_matrix
    encoder_hidden = encoder(input_variable[0], tree_matrix)

    if use_cuda:
        encoder_hidden = encoder_hidden.cuda(gpu_no)

    decoder_hiddenA = encoder_hidden
    decoder_hiddenF = encoder_hidden
    
    
    target_tree_root = target
    targetQ = Queue()
    targetQ.put(target_tree_root)
    
    nodesQ = Queue()
    root = Tree(level = 0)
    nodesQ.put(root)
     
    prev = root
    loss = 0
    loss_topo = 0
    loss_label = 0
    
    seq_len = 1
    while(not targetQ.empty()):
        target_node = targetQ.get()
        if(target_node.leftChild != None):
            seq_len += 2
            targetQ.put(target_node.leftChild)
            targetQ.put(target_node.rightChild)
        
        node = nodesQ.get()
        if node.parent == None:
            parent = Variable(torch.LongTensor([113]))
        else:
            parent = node.parent.key
            decoder_hiddenA = node.parent.hiddenA
        if node.sibling == None:
            sibling = Variable(torch.LongTensor([114]))
            if node.level != 0:
                if use_cuda:
                    decoder_hiddenF = Variable(torch.zeros(1, 1, hidden_dim).cuda(gpu_no))
                else:
                    decoder_hiddenF = Variable(torch.zeros(1, 1, hidden_dim))
        else:
            sibling = node.sibling.key
            decoder_hiddenF = node.sibling.hiddenF
            
        if use_cuda:
            parent = parent.cuda(gpu_no)
            sibling = sibling.cuda(gpu_no)

        decoder_hiddenA,decoder_hiddenF,childPred,siblingPred,output= decoder(
           decoder_hiddenA, decoder_hiddenF, parent, sibling)
        
        
        if(target_node.key in target_op):
            target_node.key = target_op[target_node.key]
        
        if use_cuda:
            target_label = Variable(torch.LongTensor([int(target_node.key)])).cuda(gpu_no)
        else:
            target_label = Variable(torch.LongTensor([int(target_node.key)]))
            
        loss_label = criterion_label(output,target_label)
        loss += loss_label

        node.key = Variable(torch.LongTensor([int(target_node.key)]))
        node.hiddenA = decoder_hiddenA
        node.hiddenF = decoder_hiddenF
        
        
        if(target_node.leftChild is not None):
            leftChild = Tree(node.level+1)
            leftChild.parent = node

            if prev.level == leftChild.level:
                leftChild.sibling = prev

            rightChild = Tree(node.level+1)
            rightChild.parent = node
            rightChild.sibling = leftChild

            prev = rightChild

            node.children.append(leftChild)
            node.children.append(rightChild)
            node.leftChild = leftChild
            node.rightChild = rightChild
            nodesQ.put(leftChild)
            nodesQ.put(rightChild)
            
            if use_cuda:
                target_topo = Variable(torch.FloatTensor([1.0]).view(1,1)).cuda(gpu_no)
            else:
                target_topo = Variable(torch.FloatTensor([1.0]).view(1,1))
        else:
            if use_cuda:
                target_topo = Variable(torch.FloatTensor([0.0]).view(1,1)).cuda(gpu_no)
            else:
                target_topo = Variable(torch.FloatTensor([0.0]).view(1,1))
        
        loss_topo = criterion_topology(childPred,target_topo)
        loss += loss_topo

    loss.backward()
    
    nn.utils.clip_grad_norm(decoder.parameters(), 5)
    nn.utils.clip_grad_norm(encoder.parameters(), 5)
    encoder_optimizer.step()
    decoder_optimizer.step() 
    
    return loss.data[0]/ target_length


# In[16]:


def trainIters(encoder, decoder, n_iters, training_pairs, print_every=1000, plot_every=100, learning_rate=0.1):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion_label = nn.NLLLoss()
    criterion_topo = nn.BCELoss()
    
    flip = True

    for iter in range(1, n_iters + 1):
        input_variable = training_pairs[0][iter -1]
        target = training_pairs[1][iter -1]
        
        loss = train(input_variable, target, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion_label,criterion_topo)
        print_loss_total += loss
        plot_losses.append(loss)
        
        
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            plot_loss_avg.append(print_loss_avg)
            print_loss_total = 0
            if flip:
                torch.save(encoder.state_dict(), './pickles/encoder_t2t_relabel_0.1.pth')
                torch.save(decoder.state_dict(), './pickles/decoder_t2t_relabel_0.1.pth')
                flip = False
            else:
                torch.save(encoder.state_dict(), './pickles/encoder_t2t_relabl_0.1.pth')
                torch.save(decoder.state_dict(), './pickles/decoder_t2t_relabel_0.1.pth')
                flip = True
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


# In[17]:


hidden_dim = 256
embedding_size = 256
gpu_no = 2

#0-108,(,)
encoder_vocab_size = 111

# 0-108,+,-,*,/, no parent, no child
decoder_vocab_size = 115

use_cuda = False
n_iters = 5000000
train_filename = "data/relabel_1/train.orig"
target_filename = "data/relabel_1/train.relabel"


target_op = {'+':'109', '-':'110','*':'111','/':'112'}


dataset = Dataset()
treeConstructor = TreeConstructor()
dataset.create_vocab(train_filename)
one_hot_dict = dataset.create_one_hot()

trees = dataset.read_trees(train_filename)
seqs = dataset.read_seqs(target_filename)
ptr_trees = [dataset.make_ptr_tree(tree, dataset.vector_dim) for tree in tqdm(trees)]
dec_trees = [treeConstructor.buildTree(seq) for seq in seqs]

training_pairs = (ptr_trees, dec_trees)

encoder_vocab_size = len(dataset.vocab.keys())
encoder1 = ChildSumTreeLSTM(encoder_vocab_size, hidden_dim)
decoder1 = DecoderDRNN(embedding_size,hidden_dim,decoder_vocab_size)

#encoder1.load_state_dict(torch.load('./pickles/encoder_drnn_relabel1.pth'))
#decoder1.load_state_dict(torch.load('./pickles/decoder_drnn_relabel1.pth'))


if use_cuda:
    encoder1 = encoder1.cuda(gpu_no)
    decoder1 = decoder1.cuda(gpu_no)


trainIters(encoder1,decoder1, n_iters, training_pairs,print_every=100)
