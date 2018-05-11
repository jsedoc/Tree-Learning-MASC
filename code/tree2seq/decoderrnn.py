
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('run', "'tree2seq_dataloader.ipynb'")


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_variable, hidden):
        output = self.embedding(input_variable).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# In[ ]:


if __name__ == '__main__':
    dataset = Dataset()
    dataset.create_vocab('data/train.orig')
    one_hot_dict = dataset.create_one_hot()

    seqs = dataset.read_seqs('data/train.orig')
    decoder = DecoderRNN(128, len(one_hot_dict.keys()))
    hidden = decoder.initHidden()
    #print(hidden)

    seq = torch.LongTensor([dataset.token_dict[x] for x in seqs[1].split()])
    input_variable = Variable(torch.LongTensor([[seq[0]]]))
    out, hidden = decoder.forward(input_variable, hidden)
    print(out)

