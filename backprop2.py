#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:50:57 2026

@author: adc20
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

#second implementation of backpropagation on a N-layer logistic neural net
#changes: I vectorised z inputs, made it variable N_alpha, and vectorised also for batch learning

def A(z):
    #ACTIVATION FUNCTION (Hinton's y(z))
    return 1/(1+np.exp(-z)) #LOGISTIC

def A_derivative(z):
    return np.exp(-z)/(1+np.exp(-z))**2

class NeuralNet():
    
    def __init__(self, biases, N=5, N_layers=2, d=4):
        #N: number of nodes per layer
        self.N = N
        self.N_layers = N_layers
        self.d = d
        self.biases = biases #list of bias vectors, one per layer, length N[alpha+1]
        self.weights = []
        if not isinstance(N, list):
            print('Initialising NN with equal layer widths.')
            for alpha in range(N_layers):
                self.weights.append(np.random.randn(self.N, self.N) * 0.5)
        else:
            print('Initialising NN with unequal layer widths (N is a list of widths).')
            for alpha in range(N_layers):
                self.weights.append(np.random.randn(self.N[alpha+1], self.N[alpha]) * 0.5)
        self.total_epochs = 0
        
    def loss_function(self, output, target):
        #l2 norm (for reporting only)
        return np.linalg.norm(output - target)
    
    def half_mse(self, output, target):
        #THE error that backprop differentiates: (1/2B) * sum (a - t)^2
        B = output.shape[0]
        return 0.5 * np.sum((output - target)**2) / B
    
    def feedforward(self, x_in, custom_weights=0):
        self.x_in = np.array(x_in)
        if custom_weights:
            weights = custom_weights
        else:
            weights = self.weights
        #lists of per-layer activities a^(alpha) and logits z^(alpha)
        activities = []
        logits = []
        logit = x_in @ weights[0].T + self.biases[0]
        activities.append(A(z = logit))
        logits.append(logit)
        for alpha in range(1,self.N_layers):
            #the output of the previous layer is the input of the next layer
            x_in = np.array(activities[alpha-1])
            logit = x_in @ weights[alpha].T + self.biases[alpha]
            activities.append(A(z = logit))
            logits.append(logit)
        self.activities = activities
        self.logits = logits
        return self.activities
    
    def backpropagate(self, targets, batch_size=None):
        #batch size is read from the data (number of rows), so it can never disagree
        #expects 2-D arrays: (cases x units)
        batch_size = self.activities[-1].shape[0]
        #initialisation: gradient dE/dyhat_i = (y_i - t_i)
        dEda = self.activities[-1] - targets
        dEdw = []
        dEdb = []
        for alpha in range(self.N_layers)[::-1]:
            #delta_i = dE/dz_i
            delta = A_derivative(self.logits[alpha]) * dEda
            #sending activities: the input for layer 0, otherwise the layer below
            prev = self.x_in if alpha == 0 else self.activities[alpha-1]
            # dE/dw_ij = delta_i * a_j, summed over the batch
            dEdw_mat = (delta.T @ prev) / batch_size
            dEdw.insert(0, dEdw_mat)
            dEdb.insert(0, delta.sum(axis=0) / batch_size) #dE/db_j = delta_j
            # dE/da for the layer below: sum over receiving nodes of w[i,j]*delta_i
            dEda = delta @ self.weights[alpha]
        return dEdw, dEdb
    
    def info(self):
        print(10*'-' + 'NeuralNet instance' + 10*'-')
        print('N, N_layers=', self.N, self.N_layers)
        print('Number of epochs trained:', self.total_epochs)
    
    def iterate_optimiser(self, x_in, target, learning_rate=0.05, batch_size=1):
        #Step 1: compute the output
        self.feedforward(x_in)
        #Step 2: backpropagate to get error sensitivities
        dEdw, dEdb = self.backpropagate(target)
        err = self.loss_function(self.activities[-1], target)
        #Step 3: update weights and biases
        new_weights = []
        new_biases = []
        for alpha in range(self.N_layers):
            new_weights.append(self.weights[alpha] - learning_rate*dEdw[alpha])
            new_biases.append(self.biases[alpha] - learning_rate * dEdb[alpha])
        self.weights = new_weights
        self.biases = new_biases
        return err
    
    def train(self, training_data, training_targets, n_epochs=4, learning_rate=0.05, \
              batch_size=None):
        self.batch_size = batch_size
        mean_errs = []
        L = len(training_data)
        if batch_size is None: 
            batch_size = L
        for epoch in range(n_epochs):
            errs = []
            for i in range(int(L / batch_size)):
                X_in = np.stack([training_data[j] for j in range(i*batch_size,(i+1)*batch_size)])
                targets = np.stack([training_targets[j] for j in range(i*batch_size,(i+1)*batch_size)])
                targets = targets.reshape(batch_size, -1)
                err = self.iterate_optimiser(X_in, targets, learning_rate=learning_rate, \
                                             batch_size=batch_size)
                errs.append(err)
            mean_errs.append(np.mean(errs))
        plt.plot(list(range(n_epochs)), mean_errs, '-', color='black')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training: mean errors at each epoch')
        plt.show()
        self.total_epochs += n_epochs
        
    def grad_check(self, x_in, target, delta_w=1e-5):
        #GOAL: compare backprop gradients with central finite differences, one parameter at a time
        #always work with 2D arrays: (cases x units)
        X = np.atleast_2d(np.array(x_in, dtype=float))
        T = np.array(target, dtype=float).reshape(X.shape[0], -1)
        
        #clean forward pass & backprop before perturbing anything
        self.feedforward(X)
        dEdw, dEdb = self.backpropagate(T)
        
        def E():
            #same error as backprop differentiates
            self.feedforward(X)
            return self.half_mse(self.activities[-1], T)
        
        #2) perturb every single weight AND bias entry, one at a time
        worst = 0.0
        for params, grads, name in [(self.weights, dEdw, 'weights'), (self.biases, dEdb, 'biases')]:
            worst_here = 0.0
            for P, G in zip(params, grads):
                for idx in np.ndindex(P.shape):
                    old = P[idx]
                    P[idx] = old + delta_w
                    err_plus = E()
                    P[idx] = old - delta_w
                    err_minus = E()
                    P[idx] = old                      #restore exactly
                    numeric = (err_plus - err_minus) / (2*delta_w)
                    analytic = G[idx]
                    rel = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-8)
                    worst_here = max(worst_here, rel)
            print(f'Worst relative error over {name}: {worst_here:.2e}')
            worst = max(worst, worst_here)
        print(f'OVERALL worst relative error: {worst:.2e}  (~1e-7 or smaller = backprop correct)')
        return worst
        
        
training_data = np.array(['00','01','10','11'])

training_targets = [sum(int(i) for i in s) % 2 for s in training_data]
#store data as ONE 2-D array (cases x inputs)
training_data = np.array([np.array(list(i), dtype=int) for i in training_data])

print('Training data:', training_data)
print('Training targets:', training_targets)

#N lists all layer widths including the input: XOR = 2 inputs, 3 hidden, 1 output
N = [2, 3, 1]
N_layers = len(N) - 1
biases = []
for alpha in range(N_layers):
    #one bias per node in the RECEIVING layer alpha+1
    #small random biases so the check isn't done at a too-symmetric point
    biases.append(np.random.randn(N[alpha+1]) * 0.1)
model = NeuralNet(N=N, N_layers=N_layers, biases=biases)

#check on the full batch of 4 cases (tests the sum over the batch too)
model.grad_check(x_in=training_data, target=training_targets, delta_w=1e-5)

#full batch averages over 4 cases, so use a larger learning rate than per-case updates
# model.train(training_data, training_targets, n_epochs=5000, learning_rate=2.0, batch_size=4)
#%%
#now see if it works on new data
print(20*'-')
print('APPLYING TRAINED MODEL TO REAL DATA')
data = np.array(['00', '01', '10', '11'])
test_targets = [sum(int(i) for i in s) % 2 for s in data]
data = list(np.array(list(i), dtype=int) for i in data)
for x_in, t in zip(data, test_targets):
    out = model.feedforward(x_in)[-1][0]
    print(x_in, 'target', t, 'output', round(out, 3), 'PASS' if round(out) == t else 'FAIL')