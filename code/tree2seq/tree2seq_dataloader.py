
# coding: utf-8

# In[1]:


# coding: utf-8

# NOTE: for sorting by value
import operator, copy

from tqdm import tqdm

import torch
import torch.utils.data as data
use_cuda = torch.cuda.is_available()

from nltk import Tree


class Dataset(data.Dataset):

    def __len__(self):
        return self.size

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [Tree.fromstring(line) for line in tqdm(f.readlines())]
        return trees
    
    def read_seqs(self, filename):
        with open(filename, 'r') as f:
            seqs = [line for line in tqdm(f.readlines())]
        return seqs
    
    def create_vocab(self, filename, max_lines = -1):
        vocab = dict()
        with open(filename, 'r') as f:
            # for token in f.read().split():
            line_num = 0
            for line in f.readlines():
                if max_lines > 0 and line_num < max_lines:
                    break
                line_num += 1
                for token in line.split():
                    if token not in vocab:
                        vocab[token] = 1
                    else:
                        vocab[token] += 1
        print(len(vocab.keys()))
        
        index = 0
        token_dict = {}
        idx2token = {}
        # from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
        sorted_vocab  = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
        for (token,_) in sorted_vocab:
            token_dict[token] = index
            idx2token[index] = token
            index += 1
    
        self.vocab = vocab
        self.token_dict = token_dict
        self.idx2token = idx2token
        
        return token_dict

    def create_one_hot(self, vector_dim = -1):
        if vector_dim < 1:
            vector_dim = len(self.token_dict.keys())
        one_hot_dict = {}
        
        for token in self.token_dict:    
            tensor = torch.zeros(1, vector_dim)
            tensor[0][self.token_dict[token]] = 1
            one_hot_dict[token] = tensor
        self.one_hot_dict = one_hot_dict
        self.vector_dim = vector_dim
        return one_hot_dict

    def fetch_one_hot(self, token_dict, token):
        return one_hot_dict[token]
    
    def make_ptr_tree(self, src_tree, vector_dim):
        tree = copy.deepcopy(src_tree)
        tree_matrix=torch.zeros(len(tree.treepositions()), vector_dim)
        (idx, tree_matrix) = self.create_pointer_tree(tree, 0, tree_matrix)
        return (tree, tree_matrix)
    
    def create_pointer_tree(self, tree, idx, tree_matrix):
        if isinstance(tree, Tree):
            for i, child in enumerate(tree):
                (idx, tree_matrix) = self.create_pointer_tree(child, idx, tree_matrix)
                if not isinstance(child, Tree):
                    tree[i] = idx - 1
            tree_matrix[idx,] = self.one_hot_dict[tree.label()]
            tree.set_label(idx)
            idx+= 1
        else:
            tree_matrix[idx,] = self.one_hot_dict[tree]
            idx += 1

        return (idx, tree_matrix) 


# In[5]:


if __name__ == '__main__':
    dataset = Dataset()
    dataset.create_vocab('data/train.orig')
    one_hot_dict = dataset.create_one_hot()
    #print(one_hot_dict['('])

    trees = dataset.read_trees('data/train.orig')
    seqs = dataset.read_seqs('data/train.orig')

    #print(trees[1])
    #print(seqs[1])


# In[9]:


#tree_matrix = torch.zeros(len(tmp.treepositions()), dataset.vector_dim)
#(idx, tree_matrix) = dataset.create_pointer_tree(tmp, 0, tree_matrix)
#(idx, tree_matrix) = dataset.create_pointer_tree(tmp)


# In[10]:


#print(dataset.one_hot_dict.keys())
#print(tmp)
#print(tree_matrix[6])


# In[11]:


#ptr_trees = [dataset.make_ptr_tree(tree) for tree in tqdm(trees)]


# In[12]:


#len(ptr_trees)


# In[13]:


#print(ptr_trees[1][0])
#print(trees[1])

