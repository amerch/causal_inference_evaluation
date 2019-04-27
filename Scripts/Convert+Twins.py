
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
import json


# In[2]:

df = pd.read_csv('../Raw_Data/TWINS/twin_pairs_X_3years_samesex.csv')
ys = pd.read_csv('../Raw_Data/TWINS/twin_pairs_Y_3years_samesex.csv')
ys = ys.drop(['Unnamed: 0'], axis=1)


# In[3]:

desc = open('../Raw_Data/TWINS/covar_desc.txt', 'r').read()
desc = eval(desc)

types = open('../Raw_Data/TWINS/covar_type.txt', 'r').read()
types = eval(types)

types['gestat10'] = 'ord'


# In[4]:

df.nunique()


# In[5]:

df['bord'] = (df['bord_0'] < df['bord_1']).astype(int)
to_remove = ['Unnamed: 0', 'Unnamed: 0.1', 'infant_id_0', 'infant_id_1',
             'brstate', 'stoccfipb', 'mplbir', 'bord_0', 'bord_1']
df = df.drop(to_remove, axis=1)

for var in to_remove + ['gestat10']:
    if var in types:
        types.pop(var)


# In[6]:

group_vars = {}
for key, value in types.items():
    group_vars[value] = group_vars.get(value, []) + [key]


# In[7]:

group_vars['cat']


# In[8]:

missing = df.isna().mean(axis=0) > 0.2


# In[9]:

max_values = (df.max(axis=0) + 1)[missing]
print (max_values.shape)


# In[10]:

mode_values = df.mode(axis=0).iloc[0][np.logical_not(missing)]
print (mode_values.shape)


# In[11]:

new_category = missing.index[missing]
mode_category = missing.index[np.logical_not(missing)]

print ("These columns are imputed using max_val + 1")
print (new_category)

print ("These columns are imputed using mode")
print (mode_category)


# In[12]:

df[new_category] = df[new_category].fillna(max_values, axis=0)
df[mode_category] = df[mode_category].fillna(mode_values, axis=0)


# In[13]:

df = pd.get_dummies(df, columns=group_vars['cat'])
print (df.shape)
print ("This is not the same as CEVAE but the closest we could get to the author's description")


# In[14]:

z = df['gestat10'].values.reshape(-1,1)
x = df.drop(['gestat10'], axis=1).values


# In[15]:

n = 5
    
w0 = 0.1  * np.random.randn(x.shape[1], n) 
wh = 5 + 0.1 * np.random.randn(1, n)
probs = expit(x @ w0 + (z / 10 - 0.1) @ wh)
t = np.random.binomial(1, probs)

ys = ys.values


# In[16]:

noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
for noise in noises:
    print (noise)
    prox = pd.get_dummies(df['gestat10']).values[:, :, np.newaxis]
    prox = np.repeat(prox, 3, 1)
    prox = np.repeat(prox, n, 2).astype(bool)
    flip = (np.random.uniform(size=prox.shape) > (1-noise))
    proxies = np.logical_xor(prox, flip).astype(int)

    x_repeat = np.repeat(x[:, :, np.newaxis], n, 2)
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
    
    mu0_train = np.zeros_like(t_train)
    mu1_train = np.zeros_like(t_train)

    mu0_test = np.zeros_like(t_test)
    mu1_test = np.zeros_like(t_test)

    for i in range(n):
        temp_x = features[:,:,i]
        temp_t = t[:,i].astype(int)
        temp_yf = ys[np.arange(ys.shape[0]), temp_t]
        temp_ycf = ys[np.arange(ys.shape[0]), 1-temp_t]
        temp_mu0 = ys[:, 0]
        temp_mu1 = ys[:, 1]
        
        x_train[:,:,i], x_test[:, :, i], t_train[:,i], t_test[:,i], yf_train[:,i], yf_test[:,i],            ycf_train[:,i], ycf_test[:,i], mu0_train[:,i], mu0_test[:,i], mu1_train[:,i], mu1_test[:,i],             = train_test_split(temp_x, temp_t, temp_yf, temp_ycf, temp_mu0, temp_mu1)

    np.savez(path + '/train.npz', x=x_train, t=t_train, yf=yf_train, ycf=ycf_train, mu1=mu1_train, mu0=mu0_train)
    np.savez(path + '/test.npz', x=x_test, t=t_test, yf=yf_test, ycf=ycf_test, mu1=mu1_test, mu0=mu0_test)


# In[ ]:



