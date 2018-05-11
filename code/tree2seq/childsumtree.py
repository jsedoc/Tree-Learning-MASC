
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('run', "'tree2seq_dataloader.ipynb'")


# In[ ]:

# from tree2seq_dataloader import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

from nltk import Tree



# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        if isinstance(tree, Tree):
            child_states = []
            for child in tree:
                child_states.append(self.forward(child, inputs))
                
            child_c, child_h = zip(* map(lambda x: x, child_states))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            state = self.node_forward(inputs[tree.label()], child_c, child_h)
        else:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            state = self.node_forward(inputs[tree], child_c, child_h)
        return state


# In[ ]:


if __name__ == '__main__':
    from tree2seq_dataloader import *
    dataset = Dataset()
    dataset.create_vocab('data/train.orig')
    one_hot_dict = dataset.create_one_hot()

    trees = dataset.read_trees('data/train.orig')
    seqs = dataset.read_seqs('data/train.orig')
    ptr_trees = [dataset.make_ptr_tree(tree, dataset.vector_dim) for tree in tqdm(trees)]

    cst = ChildSumTreeLSTM(dataset.vector_dim, 64)
    cst.forward(ptr_trees[1][0], Var(ptr_trees[1][1]))

