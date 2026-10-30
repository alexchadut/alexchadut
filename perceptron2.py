#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 00:54:17 2026

@author: adc20
"""

import numpy as np
import matplotlib.pyplot as plt


#this implements a perceptron as a binary threshold neuron, optimising the weights
#until we match the target label as well as possible. here the error is always 0 or 1
#notation: x_in is input vector, n is number of iterations, weights is vector of weights
#thresh is the threshold activation potential, y is the output data (binary label)

#SECOND implementation with the more intuitive geometric view in weight space
class Perceptron():
    def __init__(self, d, thresh=0, decimals=12):
        # self.weights = np.random.rand(d)
        self.weights = np.zeros(d+1)
        self.weights[0] -= thresh
        #d+1 because the threshold (equiv to bias) is equiv to adding a 1 in the weights
        print('INITIALISING PERCEPTRON, WEIGHTS=', self.weights)
        self.thresh = thresh
        self.decimals = decimals
        self.d = d
        
        
    def iterate_optimiser(self, x_in, target):
        x_in = np.concatenate(([1], np.array(x_in)))
        y = 1 if np.vdot(x_in, self.weights) > 0 else 0
        err = np.round(y - target, self.decimals)
        self.weights = self.weights - err * x_in
        print('TARGET (y), ERROR=\n', target, err)
        return self.weights, err
    
    def train(self, training_data, training_targets, n_epochs=4):
        for epoch in range(n_epochs):
            print(20*'-')
            print('EPOCH NUMBER:', epoch)
            errs = []
            i = 0
            for x_in, target in zip(training_data, training_targets):
                weights, err = self.iterate_optimiser(x_in, target)
                errs.append(err)
                i += 1
                print('CURRENT WEIGHTS, ERROR:', weights, err)
            plt.plot(list(range(i)), errs, '-o', color='black')
            plt.xlabel('Training data instance')
            plt.ylabel('Error')
            plt.show()
            plt.plot(range(self.d+1), weights, '-o', color='blue')
            plt.xlabel('index')
            plt.ylabel('Final weight')
            plt.show()

    def apply(self, x_in):
        y = 1 if np.vdot(np.concatenate(([1], np.array(x_in))), self.weights) > 0 else 0
        print(f'Result of trained perceptron on input {x_in}:', y)
        return y
    
#now must choose specific data to represent and learn on.
#example: numbers are either even or odd
#(Hinton calls it the N-sit parity task)
#let's see if it can predict which are even/odd

# training_data = list(np.random.randint(low=1, high=1000, size=n))
#FOR NOW: just use numbers around the same range to avoid having to treat unequal 
# binary repr sizes. convert to vector afterwards.
training_data = [20, 25, 26, 17, 21, 27]
# training_data = [20, 24, 28, 30, 19, 17, 24, 31, 22, 29]
# training_data = range(17, 32)
d = 5
#d is the length of the binary representation here
training_targets = [0 if not i % 2 else 1 for i in training_data]
training_data = list(bin(i)[2:] for i in training_data)
training_data = list(np.array(list(i), dtype=int) for i in training_data)

print('Training data:', training_data)
print('Training targets:', training_targets)

# training_data = [bin(i) for i in training_data]

model = Perceptron(d=d, thresh=0, decimals=12)

model.train(training_data, training_targets, n_epochs=10)

#now see if it works on new data
print(20*'-')
print('APPLYING TRAINED MODEL TO REAL DATA')
data = [22, 23, 24, 28, 17]
data = list(bin(i)[2:] for i in data)
data = list(np.array(list(i), dtype=int) for i in data)
print('Real data:', data)
for x_in in data:
    model.apply(x_in)

#depending heavily on the training dataset, the model learns to only place weight
#on the last digit of the binary representation, which is indeed all that matters here.

#%% now N-bit parity test in the sense of is there an even/odd no of 1s
#THE POINT: regardless of number of epochs, the errors never converge to 0, stuck around 50%.
#(see Hinton's counterexample for an even simpler 2-bit one showing perceptrons cant learn a 2-bit feature)
N = 5
from itertools import product
training_data = ['01001', '10011', '11000', '00000', '11111']
training_data = np.array(list(product([0, 1], repeat=N)))   # all 2^N inputs

d = 5
#d is the length of the binary representation here
training_targets = training_data.sum(axis=1) % 2
training_data = list(np.array(list(i), dtype=int) for i in training_data)

print('Training data:', training_data)
print('Training targets:', training_targets)

model = Perceptron(d=d, thresh=0, decimals=12)

model.train(training_data, training_targets, n_epochs=5)

#now see if it works on new data
print(20*'-')
print('APPLYING TRAINED MODEL TO REAL DATA')
data = ['11111', '00000', '01001', '11110', '11100']
data = list(np.array(list(i), dtype=int) for i in data)
print('Real data:', data)
for x_in in data:
    model.apply(x_in)
    
#The N-bit parity test can be made to work with the right features, e.g.
#if we feed the perceptron a list of N numbers (at least m bits are on) for m=1,...,N
#then the perceptron will learn this fine; these features separate the problem simply, with
#weights = (1,-1,1,-1,...)

    