
# coding: utf-8

# In[17]:


import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[78]:


train = pd.read_csv(r"C:\Users\Jameson.T.Nunnally\Desktop\digit\train.csv")
test = pd.read_csv(r"C:\Users\Jameson.T.Nunnally\Desktop\digit\test.csv")


# In[79]:


train.head()


# In[80]:


Y_train=train["label"]
Y_train.head()
X_train=train.drop(labels=["label"],axis=1)


# In[81]:


Y_train.value_counts()


# In[82]:



# Normalize the data
X_train = X_train / 255.0
test = test / 255.0


# In[83]:



# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# In[85]:


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)


# In[86]:


X_train.shape


# In[102]:


random_seed = 6


# In[103]:


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)


# In[104]:


# Some examples
g = plt.imshow(X_train[2])


# In[105]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28,1)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[106]:


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[107]:


model.fit(X_train, Y_train, epochs=10)


# In[108]:


test_loss, test_acc = model.evaluate(X_val, Y_val)

print('Test accuracy:', test_acc)


# In[109]:


predictions = model.predict(test)


# In[110]:


predictions[0]


# In[111]:


np.argmax(predictions[0])


# In[112]:


results=model.predict(test)


# In[113]:


results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[114]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv(r"C:\Users\Jameson.T.Nunnally\Desktop\digit\jims4th.csv",index=False)

