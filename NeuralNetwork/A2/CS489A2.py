# uncompyle6 version 3.2.5
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.8 |Anaconda, Inc.| (default, Dec 29 2018, 19:04:46) 
# [GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/jorchard/Dropbox/teaching/cs489_neural_nets/assignments/a2/Network.py
# Compiled at: 2019-01-30 01:44:59
# Size of source mod 2**32: 10528 bytes
import numpy as np
from IPython.display import display
from ipywidgets import FloatProgress
import time

def NSamples(x):
    """
        n = NSamples(x)
        
        Returns the number of samples in a batch of inputs.
        
        Input:
         x   is a 2D array
        
        Output:
         n   is an integer
    """
    return len(x)


def Shuffle(inputs, targets):
    """
        s_inputs, s_targets = Shuffle(inputs, targets)
        
        Randomly shuffles the dataset.
        
        Inputs:
         inputs     array of inputs
         targets    array of corresponding targets
         
        Outputs:
         s_inputs   shuffled array of inputs
         s_targets  corresponding shuffled array of targets
    """
    data = list(zip(inputs, targets))
    np.random.shuffle(data)
    s_inputs, s_targets = zip(*data)
    return (
     np.array(s_inputs), np.array(s_targets))


def Logistic(z):
    """
        y = Logistic(z)
    
        Applies the logistic function to each element in z.
    
        Input:
         z    is a scalar, list or array
    
        Output:
         y    is the same shape as z
    """
    return 1.0 / (1 + np.exp(-z))


def Logistic_p(h):
    """
        yp = Logistic_p(h)
        
        Returns the slope of the logistic function at z when h = Logistic(z).
        Note the h is the input, NOT z.
    """
    return h * (1.0 - h)


def Identity(z):
    """
        y = Identity(z)
    
        Does nothing... simply returns z.
    
        Input:
         z    is a scalar, list or array
    
        Output:
         y    is the same shape as z
    """
    return z


def Identity_p(h):
    """
        yp = Identity_p(h)
        
        Returns the slope of the identity function h.
    """
    return np.ones_like(h)


def OneHot(z):
    """
        y = OneHot(z)
    
        Applies the one-hot function to the vectors in z.
        Example:
          OneHot([[0.9, 0.1], [-0.5, 0.1]])
          returns np.array([[1,0],[0,1]])
    
        Input:
         z    is a 2D array of samples
    
        Output:
         y    is an array the same shape as z
    """
    y = []
    for zz in z:
        idx = np.argmax(zz)
        b = np.zeros_like(zz)
        b[idx] = 1.0
        y.append(b)

    y = np.array(y)
    return y


class Layer:

    def __init__(self, n_nodes, act='logistic'):
        """
            lyr = Layer(n_nodes, act='logistic')
            
            Creates a layer object.
            
            Inputs:
             n_nodes  the number of nodes in the layer
             act      specifies the activation function
                      Use 'logistic' or 'identity'
        """
        self.N = n_nodes
        self.h = []
        self.b = np.zeros(self.N)
        self.sigma = Logistic
        self.sigma_p = lambda : Logistic_p(self.h)
        if act == 'identity':
            self.sigma = Identity
            self.sigma_p = lambda : Identity_p(self.h)


class Network:

    def __init__(self, sizes, type='classifier'):
        """
            net = Network(sizes, type='classifier')
        
            Creates a Network and saves it in the variable 'net'.
        
            Inputs:
              sizes is a list of integers specifying the number
                  of nodes in each layer
                  eg. [5, 20, 3] will create a 3-layer network
                      with 5 input, 20 hidden, and 3 output nodes
              type can be either 'classifier' or 'regression', and
                  sets the activation function on the output layer,
                  as well as the loss function.
                  'classifier': logistic, cross entropy
                  'regression': linear, mean squared error
        """
        self.n_layers = len(sizes)
        self.lyr = []
        self.W = []
        if type == 'classifier':
            self.classifier = True
            self.Loss = self.CrossEntropy
            activation = 'logistic'
        else:
            self.classifier = False
            self.Loss = self.MSE
            activation = 'identity'
        for n in sizes[:-1]:
            self.lyr.append(Layer(n))

        self.lyr.append(Layer(sizes[-1], act=activation))
        for idx in range(self.n_layers - 1):
            m = self.lyr[idx].N
            n = self.lyr[idx + 1].N
            temp = (np.random.normal(size=[m, n])) / np.sqrt(m)
            self.W.append(temp)

    def FeedForward(self, x):
        """
            y = net.FeedForward(x)
        
            Runs the network forward, starting with x as input.
            Returns the activity of the output layer.
        
            All node use 
            Note: The activation function used for the output layer
            depends on what self.Loss is set to.
        """
        x = np.array(x)
        self.lyr[0].h = x
        for pre, post, W in zip(self.lyr[:-1], self.lyr[1:], self.W):
            z = (pre.h) @ (W) + post.b
            post.h = post.sigma(z)

        return self.lyr[-1].h

    def Evaluate(self, inputs, targets):
        """
            E = net.Evaluate(data)
        
            Computes the average loss over the supplied dataset.
        
            Inputs
             inputs  is an array of inputs
             targets is a list of corresponding targets
        
            Outputs
             E is a scalar, the average loss
        """
        y = self.FeedForward(inputs)
        return self.Loss(targets)

    def ClassificationAccuracy(self, inputs, targets):
        """
            a = net.ClassificationAccuracy(data)
            
            Returns the fraction (between 0 and 1) of correct one-hot classifications
            in the dataset.
        """
        y = self.FeedForward(inputs)
        yb = OneHot(y)
        n_incorrect = np.sum(yb != targets) / 2.0
        return 1.0 - float(n_incorrect) / NSamples(inputs)

    def CrossEntropy(self, t):
        """
            E = net.CrossEntropy(t)
        
            Evaluates the mean cross entropy loss between t and the activity of the top layer.
            To evaluate the network's performance on an input/output pair (x,t), use
              net.FeedForward(x)
              E = net.Loss(t)
        
            Inputs:
              t is an array holding the target output
        
            Outputs:
              E is the loss function for the given case
        """
        y = self.lyr[-1].h
        E = -np.sum(t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        return E / NSamples(t)

    def MSE(self, t):
        """
            E = net.MSE(t)
        
            Evaluates the MSE loss function using t and the activity of the top layer.
            To evaluate the network's performance on an input/output pair (x,t), use
              net.FeedForward(x)
              E = net.Loss(t)
        
            Inputs:
              t is an array holding the target output
        
            Outputs:
              E is the loss function for the given case
        """
        y = self.lyr[-1].h
        E = np.sum((y - t) ** 2) / NSamples(t)
        return E

    def BackProp(self, t, lrate=0.05):
        """
            net.BackProp(targets, lrate=0.05)
            
            Given the current network state and targets t, updates the connection
            weights and biases using the backpropagation algorithm.
            
            Inputs:
             t      an array of targets (number of samples must match the
                    network's output)
             lrate  learning rate
        """
        t = np.array(t)
        dEdz = (self.lyr[-1].h - t) / NSamples(t)
        for i in range(self.n_layers - 2, -1, -1):
            pre = self.lyr[i]
            dEdW = (pre.h.T) @ (dEdz)
            dEdb = np.sum(dEdz, axis=0)
            dEdz = dEdz @ (self.W[i].T) * pre.sigma_p()
            self.W[i] -= lrate * dEdW
            self.lyr[i + 1].b -= lrate * dEdb

    def Learn(self, inputs, targets, lrate=0.05, epochs=1):
        """
            Network.Learn(inputs, targets, lrate=0.05, epochs=1)
        
            Run through the dataset 'epochs' number of times, incrementing the
            network weights for each training sample. For each epoch, it
            shuffles the order of the samples.
        
            Inputs:
              inputs  is an array of input samples
              targets is a corresponding array of targets
              lrate   is the learning rate (try 0.001 to 0.5)
              epochs  is the number of times to go through the training data
        """
        for k in range(epochs):
            s_inputs, s_targets = Shuffle(inputs, targets)
            self.FeedForward(s_inputs)
            self.BackProp(s_targets)
# okay decompiling Network.cpython-36.pyc
