#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
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


# In[4]:


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


# In[5]:


X.shape, Y.shape


# In[91]:


g = torch.Generator().manual_seed(2147483647) # for reproducibility
embedMatrix = torch.randn((27, 2))
layer1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
layer2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [embedMatrix, layer1, b1, layer2, b2]


# In[92]:


sum(p.nelement() for p in parameters)


# In[93]:


for p in parameters:
    p.requires_grad = True


# In[9]:


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


# #### The above network is pretty good and all, but it can be improved, namely by decaying the learning rate. Let us discern what might be a good range/value to decay by.

# In[75]:


learnRatesExp = torch.linspace(-3, -0.05, 1000)
learnRates = 10**learnRatesExp


# In[76]:


lrs = []
lri = []


# In[77]:


for i in range(1000):
    
    # minibatch construct
    ix = torch.randint(0, X.shape[0], (32,))
    
    # forward pass
    emb = embedMatrix[X[ix]] # 32, 3, 2
    h = torch.tanh(emb.view(-1, 6) @ layer1 + b1) # -1 in the view means emb.shape(0)
    logits = h @ layer2 + b2
    negLossLikelihood = F.cross_entropy(logits, Y[ix])
    
    lrs.append(negLossLikelihood.item())
    lri.append(-learnRates[i])
    
    # backward pass
    for p in parameters:
        p.grad = None
    negLossLikelihood.backward()

    # update
    for p in parameters:
        p.data += -learnRates[i] * p.grad
        
print(negLossLikelihood.item())


# In[78]:


plt.plot(lri, lrs)


# #### From the above graph, we can see that a learning rate of about -0.1/-0.15 is optimal (we were right first time). It is then from here that we have the option to decay, I am not going to, however.

# In[94]:


for _ in range(10000):
    
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
    learnRate = 0.1
    for p in parameters:
        p.data += -learnRate * p.grad
        
print(negLossLikelihood.item())


# #### Now, I am going to look to split the data up into training, hyperparameter training, and testing (80%/10%/10% respectively).

# In[99]:


def build_dataset(names):
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
    return X, Y

import random
random.seed(42)
random.shuffle(names)

n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))

Xtrain, Ytrain = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[:n2])
Xtest, Ytest = build_dataset(names[n2:])


# #### Now, we take the same network model from before, but train it on the xtrain dataset.

# In[104]:


g = torch.Generator().manual_seed(2147483647) # for reproducibility
embedMatrix = torch.randn((27, 2))
layer1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
layer2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [embedMatrix, layer1, b1, layer2, b2]


# In[105]:


for p in parameters:
    p.requires_grad = True


# In[106]:


for _ in range(10000):
    
    # minibatch construct
    ix = torch.randint(0, Xtrain.shape[0], (32,))
    
    # forward pass
    emb = embedMatrix[Xtrain[ix]] # 32, 3, 2
    h = torch.tanh(emb.view(-1, 6) @ layer1 + b1) # -1 in the view means emb.shape(0)
    logits = h @ layer2 + b2
    negLossLikelihood = F.cross_entropy(logits, Ytrain[ix])
    
    # backward pass
    for p in parameters:
        p.grad = None
    negLossLikelihood.backward()

    # update
    learnRate = 0.1
    for p in parameters:
        p.data += -learnRate * p.grad
        
print(negLossLikelihood.item())


# In[110]:


# Evaluation

emb = embedMatrix[Xdev]
h = torch.tanh(emb.view(-1, 6) @ layer1 + b1)
logits = h @ layer2 + b2
loss = F.cross_entropy(logits, Ydev)
loss.item()


# #### From the fact that the loss value for the training and the development datasets is very similar, we can tell that we are *underfitting* this network; it is not yet powerful enough to overfit.
# 
# #### What this typically means, is that the network - currently 3481 parameters - is simply too small.
# 
# #### We can thus add more parameters to the network, and be quite confident that this will improve the performance of it instantly.

# In[116]:


# Changing middle layer length from 100 to 300

g = torch.Generator().manual_seed(2147483647) # for reproducibility
embedMatrix = torch.randn((27, 2))
layer1 = torch.randn((6, 300), generator=g)
b1 = torch.randn(300, generator=g)
layer2 = torch.randn((300, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [embedMatrix, layer1, b1, layer2, b2]


# In[117]:


for p in parameters:
    p.requires_grad = True


# In[118]:


for _ in range(30000):
    
    # minibatch construct
    ix = torch.randint(0, Xtrain.shape[0], (32,))
    
    # forward pass
    emb = embedMatrix[Xtrain[ix]] # 32, 3, 2
    h = torch.tanh(emb.view(-1, 6) @ layer1 + b1)
    logits = h @ layer2 + b2
    negLossLikelihood = F.cross_entropy(logits, Ytrain[ix])
    
    # backward pass
    for p in parameters:
        p.grad = None
    negLossLikelihood.backward()

    # update
    learnRate = 0.1
    for p in parameters:
        p.data += -learnRate * p.grad
        
print(negLossLikelihood.item())


# In[124]:


plt.figure(figsize=(8,8))
plt.scatter(embedMatrix[:,0].data, embedMatrix[:,1].data, s=200)
for i in range(embedMatrix.shape[0]):
    plt.text(embedMatrix[i,0].item(), embedMatrix[i,1].item(), intToChar[i], ha="center", va="center", color="white")
plt.grid('minor')


# In[ ]:




