
# coding: utf-8

# ## Convert Data
# 
# Useful notebook to visualize and convert data to npz format for general testing
# 
# Need to figure out how this was simulated in the CEVAE paper

# In[1]:

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
import os


# In[2]:

df = pd.read_csv('../Raw_Data/TWINS/twin_pairs_X_3years_samesex.csv')
ys = pd.read_csv('../Raw_Data/TWINS/twin_pairs_Y_3years_samesex.csv')
ys = ys.drop(['Unnamed: 0'], axis=1)


# In[3]:

to_include = [
    'feduc6',
    'meduc6',
    'dmar',
    'mrace',
    'frace',
    'brstate_reg',
    'dtotord_min',
    'diabetes',
    'renal',
    'alcohol',
    'tobacco',
    'adequacy',
    'gestat10',
    'pldel'
]

df = df[to_include]


# In[4]:

missing = df.isna().mean(axis=0) > 0.2


# In[5]:

max_values = (df.max(axis=0) + 1)[missing]
print (max_values.shape)


# In[6]:

mode_values = df.mode(axis=0).iloc[0][np.logical_not(missing)]
print (mode_values.shape)


# In[7]:

new_category = missing.index[missing]
mode_category = missing.index[np.logical_not(missing)]

print ("These columns are imputed using max_val + 1")
print (new_category)

print ("These columns are imputed using mode")
print (mode_category)


# In[8]:

df[new_category] = df[new_category].fillna(max_values, axis=0)
df[mode_category] = df[mode_category].fillna(mode_values, axis=0)


# In[9]:

cat = [
    'feduc6',
    'meduc6',
    'mrace',
    'frace',
    'brstate_reg',
    'adequacy',
    'pldel'
]

df = pd.get_dummies(df, columns=cat)
print (df.shape)
print ("This is not the same as CEVAE but the closest we could get to the author's description")


# In[10]:

z = df['gestat10'].values.reshape(-1,1)
x = df.drop(['gestat10'], axis=1).values


# In[11]:

n = 5 
    
w0 = 0.1  * np.random.randn(x.shape[1], n) 
wh = 5 + 0.1 * np.random.randn(1, n)
probs = expit(x @ w0 + (z / 10 - 0.1) @ wh)
t = np.random.binomial(1, probs)

ys = ys.values


# In[12]:

noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
for noise in noises:
    print (noise)
    prox = pd.get_dummies(df['gestat10']).values[:, :, np.newaxis]
    prox = np.repeat(prox, 3, 1)
    prox = np.repeat(prox, n, 2).astype(bool)
    flip = (np.random.uniform(size=prox.shape) > (1-noise))
    proxies = np.logical_xor(prox, flip).astype(int)

    x_repeat = np.repeat(x[:, :, np.newaxis], 5, 2)
    features = np.concatenate([x_repeat, proxies], axis=1)
    
    path = '../Data/Twins_%d' % (100 * noise)
    
    if not os.path.exists(path):
        os.makedirs(path)
    count = features.shape[0]
    size = int(0.75 * count)

    x_train = np.zeros((size, features.shape[1], n))
    x_test = np.zeros((count - size, features.shape[1], n))

    t_train = np.zeros((size, n))
    t_test = np.zeros((count - size, n))

    yf_train = np.zeros_like(t_train)
    ycf_train = np.zeros_like(t_train)

    yf_test = np.zeros_like(t_test)
    ycf_test = np.zeros_like(t_test)

    for i in range(n):
        temp_x = features[:,:,i]
        temp_t = t[:,i].astype(int)
        temp_yf = ys[np.arange(ys.shape[0]), temp_t]
        temp_ycf = ys[np.arange(ys.shape[0]), 1-temp_t]

        x_train[:,:,i], x_test[:, :, i], t_train[:,i], t_test[:,i], yf_train[:,i], yf_test[:,i],            ycf_train[:,i], ycf_test[:,i] = train_test_split(temp_x, temp_t, temp_yf, temp_ycf)

    np.savez(path + '/train.npz', x=x_train, t=t_train, yf=yf_train, ycf=ycf_train)
    np.savez(path + '/test.npz', x=x_test,t=t_train, yf=yf_test, ycf=ycf_train)

