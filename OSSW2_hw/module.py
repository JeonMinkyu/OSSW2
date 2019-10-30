import numpy as np
from collections import OrderedDict
from layer import *
from function import * 

class fc_module:
    def __init__(self, input_size, hidden_size, output_size, loss_type='cross_entropy', act='relu', weight_init_std=0.01):
        self.loss_type = loss_type
        self.params = {}
        self.hidden_size = hidden_size
        
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size[0])
        self.params['b1'] = np.zeros(hidden_size[0])
        
        for i in range(1,len(hidden_size)):
            self.params['W'+str(i+1)] = weight_init_std * np.random.randn(hidden_size[i-1],hidden_size[i])
            self.params['b'+str(i+1)] = np.zeros(hidden_size[i])
        
        self.params['W'+str(len(hidden_size)+1)] = weight_init_std * np.random.randn(hidden_size[len(hidden_size)-1], output_size)
        self.params['b'+str(len(hidden_size)+1)] = np.zeros(output_size)

        self.layers = OrderedDict()
        
        if act == 'relu':
            for i in range(1,len(hidden_size)+1):
                self.layers['Affine'+str(i)] = Affine(self.params['W'+str(i)], self.params['b'+str(i)])
                self.layers['act'+str(i)] = Relu()
                
        if act == 'sigmoid':
            for i in range(1,len(hidden_size)+1):
                self.layers['Affine'+str(i)] = Affine(self.params['W'+str(i)], self.params['b'+str(i)])
                self.layers['act'+str(i)] = Sigmoid()

        self.layers['Affine'+str(len(hidden_size)+1)] = Affine(self.params['W'+str(len(hidden_size)+1)], self.params['b'+str(len(hidden_size)+1)])
        
        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        
        y = self.predict(x)
        
        return self.last_layer.forward(y,t,self.loss_type)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim!=1:
            t = np.argmax(t, axis=1)
            
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())

        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db
        
        return grads