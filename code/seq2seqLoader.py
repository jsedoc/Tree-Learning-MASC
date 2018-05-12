import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import re


# In[76]:

class S2SDataLoader:
    def __init__(self,train_file,target_file):
        self.pairs = []
        self.parse_file(train_file,target_file)
        
    def get_data(self):
        return self.pairs
    
    def parse_file(self,train_file,target_file):
        with open(train_file) as train, open(target_file) as target: 
            for train_line, target_line in zip(train, target):
                train_words = train_line.strip().split()
                train_words = [w.replace('(', '109').replace(')','110')
                         for w in train_words]
 
                train_words = [int(i) for i in train_words]
    
                #SOS
                train_words.insert(0,111)
                #EOS 
                train_words.append(112)
                
                target_words = target_line.strip().split()
                target_words = [w.replace('(', '113').replace(')','114')
                         .replace('+','109').replace('-','110')
                         .replace('*','111').replace('/','112') 
                         for w in target_words]
                
                
                target_words = [int(i) for i in target_words]
                #SOS
                target_words.insert(0,115)
                #EOS 
                target_words.append(116)
                
                pair = []
                pair.append(train_words)
                pair.append(target_words)
                
                self.pairs.append(pair)
