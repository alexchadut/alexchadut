#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:50:57 2026
@author: adc20

"""

#backprop4:  mean squared error or cross entropy implemented
#LEARNING MNIST DATASET OF NUMBER IMAGES VIA BACKPROP AND MINI-BATCHES

import numpy as np
import matplotlib.pyplot as plt

def A(z):
    #ACTIVATION FUNCTION (Hinton's y(z))
    return 1/(1+np.exp(-z)) #LOGISTIC

def A_derivative(z):
    return np.exp(-z)/(1+np.exp(-z))**2

class NeuralNet():
    
    def __init__(self, N, loss='mse'):
        #N: list of ALL layer widths including the input, e.g. [2, 3, 1]
        self.N = N
        self.N_layers = len(N) - 1
        self.loss = loss                      #'mse' or 'ce'
        self.weights = []
        self.biases = []
        for alpha in range(self.N_layers):
            self.weights.append(np.random.randn(N[alpha+1], N[alpha]) * (1 / np.sqrt(N[alpha])))
            self.biases.append(np.random.randn(N[alpha+1]) * 0.1)
        self.total_epochs = 0
        
    def error(self, output, target):
        #THE error that backprop differentiates, averaged over the batch
        B = output.shape[0]
        if self.loss == 'mse':
            return 0.5 * np.sum((output - target)**2) / B
        else:
            eps = 1e-12
            return -np.sum(target*np.log(output+eps) + (1-target)*np.log(1-output+eps)) / B
    
    def feedforward(self, x_in):
        self.x_in = np.array(x_in)
        activities = []
        logits = []
        a = self.x_in
        for alpha in range(self.N_layers):
            logit = a @ self.weights[alpha].T + self.biases[alpha]
            a = A(logit)
            activities.append(a)
            logits.append(logit)
        self.activities = activities
        self.logits = logits
        return self.activities
    
    def backpropagate(self, targets):
        #expects 2-D arrays: (cases x units); batch size read from the data
        batch_size = self.activities[-1].shape[0]
        dEdw = []
        dEdb = []
        dEda = 0
        for alpha in range(self.N_layers)[::-1]:
            if alpha == self.N_layers - 1 and self.loss == 'ce':
                #cross-entropy + logistic: the A' factor cancels (L3 exercise 2)
                delta = self.activities[-1] - targets
            elif alpha == self.N_layers - 1:
                delta = A_derivative(self.logits[alpha]) * (self.activities[-1] - targets)
            else:
                delta = A_derivative(self.logits[alpha]) * dEda
            prev = self.x_in if alpha == 0 else self.activities[alpha-1]
            dEdw.insert(0, (delta.T @ prev) / batch_size)
            dEdb.insert(0, delta.sum(axis=0) / batch_size)
            dEda = delta @ self.weights[alpha]
        return dEdw, dEdb
    
    def iterate_optimiser(self, x_in, target, learning_rate):
        self.feedforward(x_in)
        dEdw, dEdb = self.backpropagate(target)
        err = self.error(self.activities[-1], target)
        for alpha in range(self.N_layers):
            self.weights[alpha] = self.weights[alpha] - learning_rate*dEdw[alpha]
            self.biases[alpha] = self.biases[alpha] - learning_rate*dEdb[alpha]
        return err
    
    def train(self, X, T, n_epochs, learning_rate, batch_size=None):
        print(f'Training model for {n_epochs} epochs...')
        #mini-batch SGD; batch_size=None -> full batch. Returns mean error per epoch.
        X = np.asarray(X, dtype=float)
        T = np.array(T, dtype=float).reshape(len(X), -1)
        L = len(X)
        if batch_size is None:
            batch_size = L
        errs = []
        for epoch in range(n_epochs):
            perm = np.random.permutation(L)            #reshuffle every epoch
            batch_errs = []
            for k in range(0, L, batch_size):          #last partial batch kept
                idx = perm[k:k+batch_size]
                batch_errs.append(self.iterate_optimiser(X[idx], T[idx], learning_rate))
            errs.append(np.mean(batch_errs))
        self.total_epochs += n_epochs
        print('MEAN ERRORS:', errs)
        # plt.plot(range(n_epochs), errs, color='black')
        # plt.xlabel('Epoch')
        # plt.ylabel('Mean err')
        # plt.title('Training')
        # plt.show()
        return np.array(errs)
    
    def predict(self, X):
        return self.feedforward(X)[-1]
        
    def grad_check(self, X, T, h=1e-5):
        X = np.atleast_2d(np.array(X, dtype=float))
        T = np.array(T, dtype=float).reshape(X.shape[0], -1)
        #1 clean forward pass + backprop BEFORE perturbing anything
        self.feedforward(X)
        dEdw, dEdb = self.backpropagate(T)
        def E():
            self.feedforward(X)
            return self.error(self.activities[-1], T)
        #2 perturb every weight and bias entry, one at a time
        worst = 0.0
        for params, grads in [(self.weights, dEdw), (self.biases, dEdb)]:
            for P, G in zip(params, grads):
                for idx in np.ndindex(P.shape):
                    old = P[idx]
                    P[idx] = old + h;  Ep = E()
                    P[idx] = old - h;  Em = E()
                    P[idx] = old                      #restore exactly
                    numeric = (Ep - Em) / (2*h)
                    rel = abs(numeric - G[idx]) / max(abs(numeric), abs(G[idx]), 1e-8)
                    worst = max(worst, rel)
        return worst

def bits(n_bits):
    #all 2^n bit-strings as a 2-D array, and their parity targets
    X = np.array([[int(b) for b in format(i, f'0{n_bits}b')] for i in range(2**n_bits)])
    return X, X.sum(axis=1) % 2

def n_correct(model, X, T):
    return int(np.sum(np.round(model.predict(X)[:, 0]) == T))

#%% EXERCISES
def plot_image(X, index):
    plt.subplot(1, 2, 1)
    plt.imshow(X[index].reshape(28, 28), cmap='gray')
    plt.title(f'Index {index}')
    plt.axis('off')
    plt.show()

import sklearn
from sklearn.datasets import load_digits

mnist = sklearn.datasets.fetch_openml('mnist_784', version=1, as_frame=False)
data, target = mnist['data'], mnist['target']
index1, index2 = 0, 60000
L = len(data)
X_train = data[index1:index2]
X_test = data[index2:]
y_train = target[index1:index2]
y_test = target[index2:]

plot_image(X_test,0)

X_train = X_train.astype(np.float64) / 255
X_test = X_test.astype(np.float64) / 255


T_train = np.zeros((index2-index1, 10))
for i in range(index2-index1):
    for j in range(10):
        if j == int(y_train[i]):
            T_train[i,j] += 1
T_test = np.zeros((L - index2, 10))
for i in range(L - index2):
    for j in range(10):
        if j == int(y_test[i]):
            T_test[i,j] += 1
#in our convention here, the first layer IS the input so must have same dimension
#as the input. We only have one hidden layer of 100 nodes here.
model = NeuralNet([784, 100, 10], loss='ce') #THIS ONE REQCHES 97%
# model = NeuralNet([784, 10], loss='ce') #THIS ONE REQCHES 92%

# model.feedforward(X_train[:100])
# model.train(X_train, T_train, n_epochs=10, learning_rate=0.5, batch_size=100)
n_epochs = 10
for i in range(n_epochs):
    model.train(X_train, T_train, n_epochs=1, learning_rate=0.5, batch_size=100)
    #test accuracy
    results = model.predict(X_test)
    indices = [np.argmax(result) for result in results]
    diff = [indices[i] - int(y_test[i]) for i in range(L-index2)]
    print('ACCURACY:', 1 - (len(np.where(diff)[0])  / (L - index2)))
#LESSON 1: adding a hidden layer closes a gap from 92% to 97%
#LESSON 2: then network generalises to images it has never seen, which we did not try for XOR parity

#LESSON 3: what did the network learn?
incoming_hidden_units_weights = model.weights[0]          # shape (100, 784): one row per hidden unit
detectors = [w.reshape((28, 28)) for w in incoming_hidden_units_weights]

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for ax, d in zip(axes.flat, detectors):
    lim = np.abs(d).max()                                  # symmetric colour scale per unit
    ax.imshow(d, cmap='RdBu_r', vmin=-lim, vmax=lim)       # red = positive weight, blue = negative
    ax.axis('off')
plt.suptitle('Incoming weights of the 100 hidden units')
plt.tight_layout()
plt.show()

