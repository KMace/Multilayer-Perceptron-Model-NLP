#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


names = open("names.txt", "r").read().split()


# In[3]:


# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(names))))
charToInt = {s:i+1 for i, s in enumerate(chars)}
charToInt['.'] = 0

intToChar = {i:s for s, i in charToInt.items()}


# In[18]:


# build the dataset

block_size = 3
X, Y = [], []
for name in names:
    context = [0] * block_size
    
    for char in name + '.':
        index = charToInt[char]
        X.append(context)
        Y.append(index)
        context = context[1:] + [index]
        
X = torch.tensor(X)
Y = torch.tensor(Y)


# In[19]:


X.shape, Y.shape


# In[20]:


g = torch.Generator().manual_seed(2147483647) # for reproducibility
embedMatrix = torch.randn((27, 2))
layer1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
layer2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [embedMatrix, layer1, b1, layer2, b2]


# In[21]:


sum(p.nelement() for p in parameters)


# In[22]:


for p in parameters:
    p.requires_grad = True


# In[28]:


for _ in range(1000):
    
    # minibatch construct
    ix = torch.randint(0, X.shape[0], (32,))
    
    # forward pass
    emb = embedMatrix[X[ix]] # 32, 3, 2
    h = torch.tanh(emb.view(-1, 6) @ layer1 + b1) # -1 in the view means emb.shape(0)
    logits = h @ layer2 + b2
    negLossLikelihood = F.cross_entropy(logits, Y[ix])
    
    # backward pass
    for p in parameters:
        p.grad = None
    negLossLikelihood.backward()

    # update
    for p in parameters:
        p.data += -0.1 * p.grad
        
print(negLossLikelihood.item())

