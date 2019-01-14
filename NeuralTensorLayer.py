
# coding: utf-8

# In[1]:


from keras import optimizers
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

'''
import warnings
warnings.filterwarnings('ignore')
print ('Libraries Loaded')
'''

import os
from importlib import reload # for python 3

USE_THEANO = True #False

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

if USE_THEANO: set_keras_backend("theano")


# ![alt text](img1.png "The Function modelled by the Neural Tensor Network")
# 
# This is the function modelled by the neural tensor layer where 
# 
# > * f is a standard nonlinearity applied element-wise, 
# 
# > * W<sup>[1:k]</sup><sub>R</sub>∈ R<sup>d×d×k</sup> is a tensor and the bilinear tensor product e<sup>T</sup><sub>1</sub>W<sup>[1:k]</sup><sub>R</sub>e2<sub>2</sub> results in a vector h ∈ R<sup>k</sup>, where each entry is computed by one slice i = 1, . . . , k of the tensor: hi = e<sup>T</sup><sub>1</sub>W<sup>[i]</sup><sub>R</sub>e2<sub>2</sub>. 
# 
# > * The other parameters for relation R are the standard form of a neural network: V<sub>R</sub>∈ R<sup>kx2d</sup>and U ∈ R<sup>k</sup>, b<sub>R</sub> ∈ R<sup>k</sup>.
# 
# Four methods are required to be included in the class which will help to model the layer in keras :<br><br>
# * **__init()__** - This is used to initialise the layer.
# 
# * **build(self, input_shape)** - Initialise the tensor variables and set the variables to be trained.
# 
# * **call(self, x, mask=None)** - The forward pass operation is implemented here.
# 
# * **get_output_shape_for(self, input_shape)** - Used to get the output shape before the network actually runs to help the building of the graph.
# 
# ![alt text](img2.png "The Function modelled by the Neural Tensor Network")
# 
# **This is the block architecture of the model.**

# In[5]:


class NeuralTensorLayer(Layer):
    
    def __init__(self, output_dim, input_dim, activation= None):
        super(NeuralTensorLayer, self).__init__()
        self.output_dim = output_dim #The k in the formula
        self.input_dim = input_dim   #The d in the formula
        self.activation = activation #The f function in the formula
        
    def build(self, input_shape):
        #The initialisation parameters
        self.mean = 0.0 
        self.stddev = 1.0
        dtype = 'float32'
        self.seed = 1
        
        #The output and the inut dimension
        k = self.output_dim
        d = self.input_dim
        
        #Initialise the variables to be trained. The variables are according to the
        #function defined.
        self.W = K.variable(K.random_normal((k,d,d), self.mean, self.stddev,
                               dtype=dtype, seed=self.seed))
        self.V = K.variable(K.random_normal((2*d,k), self.mean, self.stddev,
                               dtype=dtype, seed=self.seed))
        self.b = K.zeros((self.input_dim,))
        
        #Set the variables to be trained.
        self.trainable_weights = [self.W, self.V, self.b]

    def call(self, inputs):
        
        #Get Both the inputs
        e1 = inputs[0]
        e2 = inputs[1]
        
        #Get the batch size
        batch_size = K.shape(e1)[0]
        
        #The output and the inut dimension
        k = self.output_dim
        d = self.input_dim

        #The first term in the function which is the bilinear product is calculated here.
        first_term_k = [K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1)]
        for i in range(1, k):
            temp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
            first_term_k.append(temp)
        first_term = K.reshape(K.concatenate(first_term_k, axis=0), (batch_size, k))

        #The second term in the function is calculated here.
        second_term = K.dot(K.concatenate([e1,e2]), self.V)
        
        #Sum of the two terms to get the final function
        z =  first_term + second_term
        
        #The activation is selected here
        if (self.activation == None):
            return z
        elif (self.activation == 'tanh'):
            return K.tanh(z)
        elif (self.activation == 'relu'):
            return K.relu(z)
        else :
            print ('Activation not found')
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)


# In[6]:


# Dummy training data
x_train1 = np.random.random((1000, 300))
x_train2 = np.random.random((1000, 300))
y_train = np.random.random((1000, 1))

# Dummy validation data
x_val1 = np.random.random((100, 300))
x_val2 = np.random.random((100, 300))
y_val = np.random.random((100, 1))

print ('Shape of Training Data: ', x_train1.shape, x_train2.shape, y_train.shape)
print ('Shape of Validation Data', x_val1.shape, x_val2.shape, y_val.shape)


# In[7]:


#Here Define the model
vector1 = Input(shape=(300,), dtype='float32')
vector2 = Input(shape=(300,), dtype='float32')
BilinearLayer = NeuralTensorLayer(output_dim=32, input_dim=300, \
                                  activation= 'relu')([vector1, vector2])

#The g or the output of the modelled function.
g = Dense(output_dim=1)(BilinearLayer)
model = Model(input=[vector1, vector2], output=[g])

#Compile the model
adam = optimizers.adam(.001)
model.compile( loss='mean_squared_error', optimizer=adam)
#The summary of the model.
model.summary()


# In[8]:


model.fit([x_train1, x_train2], y_train,
          batch_size=64, epochs=5,
          validation_data=([x_val1, x_val2], y_val))

