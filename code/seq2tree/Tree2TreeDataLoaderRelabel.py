
# coding: utf-8

# In[2]:

import sys, os, re

try:
    from TreeConstructor import *
except:
    from seq2tree.TreeConstructor import *



# In[8]:


class Tree2TreeDataLoader:
    def __init__(self,train_file,target_file):
        self.pairs = []
        self.parse_file(train_file,target_file )
        
    def get_data(self):
        return self.pairs
    
    def parse_file(self,train_file,target_file):

        with open(train_file) as train, open(target_file) as target: 
            for train_line, target_line in zip(train, target):
                  # USE FOR INFIX/PREFIX INPUT 
                train_words = train_line.strip().split()
                train_words = [w.replace('(', '109').replace(')','110')
                         for w in train_words]
 
                train_words = [int(i) for i in train_words]
    
                #Remove underlying line if you dont want to reverse input
#                 train_words = train_words[::-1]
    
                  # USE FOR BFS
#                 treeConstructor_inp = TreeConstructor()
#                 train_bfs = treeConstructor_inp.buildTree(train_line)
#                 train_bfs_str = treeConstructor_inp.buildBFSStr(train_bfs)
#                 train_bs_words = train_bfs_str.strip().split()
#                 train_words = [int(i) for i in train_bs_words]
                
                
                treeConstructor = TreeConstructor()
                
                pair = []
                pair.append(train_words)
                pair.append(treeConstructor.buildTree(target_line))
                self.pairs.append(pair)

                


# # In[9]:


# train_filename = "/data2/t2t/synth_data/relabel/train.orig"
# target_filename = "/data2/t2t/synth_data/relabel/train.interior_relabel"
# data_loader = Tree2TreeDataLoader(train_filename,target_filename)
# training_pairs = data_loader.get_data()


# # In[13]:


# print(training_pairs[0])
# print(training_pairs[0][1].key)
# print(training_pairs[0][1].leftChild.key)
# print(training_pairs[0][1].rightChild.key)
# print(training_pairs[0][2])

