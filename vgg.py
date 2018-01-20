
# coding: utf-8

# In[4]:





# In[1]:


import keras


# In[2]:


from keras.models import Sequential


# In[3]:


from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout, Lambda


# In[4]:


from keras.preprocessing import image


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


from keras import backend as K


# In[8]:


from keras.utils.data_utils import get_file
import json


# In[9]:


FILES_PATH = 'http://files.fast.ai/models/'; CLASS_FILE='imagenet_class_index.json'
# Keras' get_file() is a handy function that downloads files, and caches them for re-use later
fpath = get_file(CLASS_FILE, FILES_PATH+CLASS_FILE, cache_subdir='models')
with open(fpath) as f: class_dict = json.load(f)
# Convert dictionary with string indexes into an array
classes = [class_dict[str(i)][1] for i in range(len(class_dict))]




def ConvBlock(layers, model, filters):
    for i in range(layers):
        model.add(Convolution2D(filters, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    


# In[12]:


def FCBlock(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))


# In[13]:


import numpy as np


# In[14]:


vgg_mean = np.array([123.68, 116.779, 103.939])


# In[44]:

#Use this for theano dim ordering
#def vgg_preprocess(x):
#    x = x - vgg_mean
#    return x[:,::-1]

#tensorflow dim ordering
def vgg_preprocess1(x):
    x = x - vgg_mean
    return x


# In[69]:


def vgg_16():
    model = Sequential()
	#Using TF dim ordering
    model.add(Lambda(vgg_preprocess1, input_shape=(224,224,3)))
    ConvBlock(2, model, 64)
    ConvBlock(2, model, 128)
    ConvBlock(3, model, 256)
    ConvBlock(3, model, 512)
    ConvBlock(3, model, 512)
    
    model.add(Flatten())
    FCBlock(model)
    FCBlock(model)
    model.add(Dense(1000, activation='softmax'))
    return model

def getVGGModel():
	vggmodel = vgg_16()
	vggmodel.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
	return vggmodel

