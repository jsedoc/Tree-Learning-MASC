
# coding: utf-8

# In[15]:
from queue import *


class Target:
    def __init__(self,rootObj=None):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None
        self.children = [] 


# In[16]:


class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)


# In[25]:


class TreeConstructor:
    def buildTree(self,treeStr):
        treelist = treeStr.split()
        
        rootNode = Target()
        
        treeStack = Stack()
        treeStack.push(treelist[0])
        
        i = 1   
        while treeStack.size() > 0:
            if treelist[i] == ')':
                right = treeStack.pop()
                root = treeStack.pop()
                left = treeStack.pop()
                
                #Popping the ( symbol
                treeStack.pop()
                
                root.leftChild = left
                root.rightChild = right
                
                root.children.append(left)
                root.children.append(right)
                
                treeStack.push(root)
                    
            else:
                treeStack.push(Target(treelist[i]))
            i += 1
            
            if i == len(treelist) and treeStack.size() == 1:
                rootNode = treeStack.pop()
                
        return rootNode
    
    def buildBFSStr(self,rootNode):
        nodeQ = Queue()
        nodeQ.put(rootNode)
        bfsStr = ""
        while(not nodeQ.empty()):
            
            node = nodeQ.get()
            if(node == '109'):
                bfsStr+= " "+ '109'
            else:
                bfsStr+= " "+node.key
                if(node.leftChild is not None):
                    nodeQ.put(node.leftChild)
                    nodeQ.put(node.rightChild)
                else:
                    nodeQ.put('109')
                    nodeQ.put('109')
        return bfsStr
                
            
            
        
        
        


# # In[26]:
if __name__ == '__main__':
    c = TreeConstructor()
    rootNode = c.buildTree("( ( 86 - ( ( 93 - 54 ) * 92 ) ) * ( ( 39 * 17 ) * ( 74 / 49 ) ) )")
    bfsStr = c.buildBFSStr(rootNode)
    print(bfsStr)
        


# # In[43]:


# print(rootNode.key)
# print(rootNode.leftChild.key)
# print(rootNode.leftChild.leftChild.key)
# print(rootNode.leftChild.rightChild.key)
# print(rootNode.leftChild.rightChild.leftChild.key)
# print(rootNode.leftChild.rightChild.rightChild.key)


# print(rootNode.rightChild.key)
# print(rootNode.rightChild.rightChild.key)
# print(rootNode.rightChild.leftChild.key)
# print(rootNode.rightChild.leftChild.rightChild.key)
# print(rootNode.rightChild.leftChild.leftChild.key)
# print(rootNode.rightChild.leftChild.leftChild.leftChild.key)
# print(rootNode.rightChild.leftChild.leftChild.rightChild.key)
# print(rootNode.rightChild.leftChild.leftChild.rightChild.rightChild)

