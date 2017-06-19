
# coding: utf-8

# In[191]:

from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Flatten, Reshape, Activation, Lambda, Permute
from keras.layers import Input
from keras.layers.merge import add
from keras.initializers import RandomNormal
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.utils import plot_model
from IPython.core.display import Image, display
import numpy as np


# In[243]:

DIM = 200
SEQ_LEN = 32
RANDOM_DIM = 128
VOCAB_SIZE = 128


# In[244]:

def ResBlock(name):
    inputs = Input(shape=(SEQ_LEN, DIM))
    block = Sequential(name=name+'_res')
    block.add(Activation('relu', input_shape=(SEQ_LEN, DIM)))
    block.add(Conv1D(DIM, 5, padding='same', name=name+"_cov1", input_shape=(SEQ_LEN, DIM)))
    block.add(Activation('relu'))
    block.add(Conv1D(DIM, 5, padding='same', name=name+"_cov2", input_shape=(SEQ_LEN, DIM)))
    merged = Lambda(lambda q: q[0]*0.3 + q[1])([block(inputs), inputs])    
    block_model = Model(inputs=inputs, outputs=merged)
    return block_model


# In[245]:

def Generator():
    generator = Sequential()
    generator.add(Dense(SEQ_LEN*DIM, input_shape=(RANDOM_DIM,)))
    generator.add(Reshape((SEQ_LEN, DIM)))
    for i in range(5):
        generator.add(ResBlock('Generator.{0:d}'.format(i)))
    generator.add(Conv1D(VOCAB_SIZE, 5, padding='same'))
    generator.add(Activation('softmax'))
    return generator


# In[262]:

def Discriminator():
    discriminator = Sequential()
    discriminator.add(Conv1D(DIM, 5, padding='same', input_shape=(SEQ_LEN, VOCAB_SIZE)))
    for i in range(5):
        discriminator.add(ResBlock('Generator.{0:d}'.format(i)))
    discriminator.add(Reshape((SEQ_LEN*DIM,)))   
    discriminator.add(Dense(1, activation='softmax'))
    return discriminator


# In[259]:

g = Generator()
print(g.output_shape)
g.summary()


# In[263]:

d = Discriminator()
print(d.output_shape)
d.summary()


# In[ ]:




# In[167]:

t = ResBlock('test', 100)


# In[168]:

t.summary()


# In[169]:

pred = t.predict(np.ones((100, SEQ_LEN, DIM)))


# In[114]:

pred[0]


# In[49]:

plot_model(t, to_file='model.png')


# In[53]:

display(Image('model.png', width=100, unconfined=True))


# In[ ]:




# In[ ]:

class Discriminator(object):
    def __init__(self):
        self.x_dim = 784
        self.name = 'mnist/dcgan/discriminator'
        self.initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
        self.regularizer = regularizers.l2(2.5e-5)

    def __call__(self):
        model = Sequential()
        model.add(Reshape((28, 28, 1), input_shape=(784,)))
        # Convolution Layer 1
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2),             kernel_initializer=self.initializer))
        model.add(LeakyReLU())

        # Convolution Layer 2
        model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2),             kernel_initializer=self.initializer))
        model.add(LeakyReLU())

        # Batch Normalization
        model.add(BatchNormalization())

        # Flatten the input
        model.add(Flatten())

        # Dense Layer
        model.add(Dense(1024, kernel_initializer=self.initializer))
        model.add(LeakyReLU())

        # Batch Normalization
        model.add(BatchNormalization())

        # To the output that has two classes
        model.add(Dense(2, activation='softmax'))

return model


# In[ ]:



