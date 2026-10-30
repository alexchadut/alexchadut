#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:50:57 2026

@author: adc20
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

#first implementation of backpropagation on a N-layer logistic neural net

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
        self.biases = biases #should be a (N_layer,N) dimensional matrix
        #for each node, we have a weight vector of some dimension d
        self.weights = []
        for alpha in range(N_layers):
            #at each layer alpha we have the matrix w^(alpha)_ij
            # self.weights.append(np.random.random((self.N, self.N)))
            self.weights.append(np.random.randn(self.N, self.N) * 0.5)
        self.total_epochs = 0
        
    def loss_function(self, output, target):
        #l2 norm
        return np.linalg.norm(output - target)
        #cross-entropy
        # return ...
    
    def feedforward(self, x_in, custom_weights=0):
        self.x_in = np.array(x_in)
        #given input, go through the whole network to compute final output
        #initialise outputs at first layer
        if custom_weights:
            weights = custom_weights
        else:
            weights = self.weights
        #matrix of activities a^(alpha)_i
        activities = np.zeros((self.N_layers, self.N))
        logits = np.zeros((self.N_layers, self.N))
        # x_ij = np.zeros((self.N_layers, self.N))
        for i in range(self.N):
            logit = self.biases[0,i] + sum(x_in[j]*weights[0][i,j] for j in range(self.N))
            activities_i = A(z = logit)
            activities[0,i] += activities_i
            logits[0,i] += logit
        for alpha in range(1,self.N_layers):
            #recursively convert the output of the previous layer as the input of the next layer
            x_in = np.array(activities[alpha-1])
            for i in range(self.N):
                logit = self.biases[alpha,i] + sum(x_in[j]*weights[alpha][i,j] for j in range(self.N))
                #what is the input to a neuron in the next layer? is it a vector of outputs of the previous?
                #then this becomes
                activity_i = A(z = logit)
                activities[alpha,i] += activity_i
                logits[alpha,i] += logit
        # print('activities = \n', activities)
        # print('logits = \n', logits)

        self.activities = activities
        self.logits = logits
        # print('Final outputs:', self.activities[-1])
        # print('Stored all nodal activities as self.activities.')
        return self.activities
    
    def backpropagate(self, targets):
        #GOAL: obtain sensitivities of the error to all weight coefficients
        #initialisation: gradient dE/dyhat_i = (y_i - t_i)
        dEda = self.activities[-1] - targets
        dEdw = []
        dEdb = []
        for alpha in range(self.N_layers)[::-1]:
            #delta_i = dE/dz_i
            delta = A_derivative(self.logits[alpha]) * dEda
            #sending activities: the input for layer 0, otherwise the layer below
            prev = self.x_in if alpha == 0 else self.activities[alpha-1]
            # dE/dw_ij = delta_i * a_j
            dEdw_mat = np.outer(delta, prev)
            dEdw.insert(0, dEdw_mat)
            dEdb.insert(0, delta) #dE/db_j = delta_j
            # dE/da_i for the layer below: sum over receiving nodes i of w[i,j]*delta_i
            dEda = self.weights[alpha].T @ delta
        return dEdw, dEdb
    
    def info(self):
        print(10*'-'+'NeuralNet instance'+10*'-')
        print('N, N_layers=', self.N, self.N_layers)
        print('Number of epochs trained:', self.total_epochs)
    
    def plotter():
        #plot useful things here eg distribution of weights, of activities...
        return None
    
    def iterate_optimiser(self, x_in, target, learning_rate=0.05):
        #in this algorithm weights undergo gradient descent.
        
        #FIRST NAIVE VERSION: perturb a weight randomly, see if the output improves
        # output, _ = self.compute_output(x_in)
        # index1 = np.random.randint(0, self.N_layers)
        # index2 = np.random.randint(0, self.N)
        # epsilon = np.random.random_sample()
        # perturbed_weights = self.weights
        # perturbed_weights[(index1,index2)] += epsilon
        # perturbed_output = self.compute_output(x_in, custom_weights=perturbed_weights)
        # loss, loss_pert = self.loss_function(output), self.loss_function(perturbed_output)
        # print('Loss function before / after perturbation:', loss, loss_pert)
        # if loss_pert < loss:
        #     self.weights = perturbed_weights
            
        #SECOND VERSION: backpropagation
        #Step 1: compute the output
        self.feedforward(x_in)
        #Step 2: backpropagate to get error sensitivities
        dEdw, dEdb = self.backpropagate(target)
        err = self.loss_function(self.activities[-1], target)
        #Step 3: update weights
        new_weights = []
        for alpha in range(self.N_layers):
            new_weights.append(self.weights[alpha] - learning_rate*dEdw[alpha])
        self.weights = new_weights
        self.biases = self.biases - learning_rate * np.array(dEdb)
        return err
    
    def train(self, training_data, training_targets, n_epochs=4, learning_rate=0.05):
        mean_errs = []
        for epoch in range(n_epochs):
            print(20*'-')
            print('EPOCH NUMBER:', epoch)
            errs = []
            i = 0
            for x_in, target in zip(training_data, training_targets):
                err = self.iterate_optimiser(x_in, target, learning_rate=learning_rate)
                errs.append(err)
                i += 1
                print('CURRENT ERROR:', err)
            mean_errs.append(np.mean(errs))
        plt.plot(list(range(n_epochs)), mean_errs, '-o', color='black')
        plt.xlabel('Training data instance')
        plt.ylabel('Error')
        plt.title('Training: mean errors at each epoch')
        plt.show()
        self.total_epochs += n_epochs
        
training_data = np.array(['01001', '10011', '11000', '00000', '11111'])
training_data = np.array(['00','01','10','11'])

#d is the length of the binary representation here
training_targets = [sum(int(i) for i in s) % 2 for s in training_data]
training_data = list(np.array(list(i), dtype=int) for i in training_data)

print('Training data:', training_data)
print('Training targets:', training_targets)

N_layers, N = 2, 2
model = NeuralNet(N=N, N_layers=N_layers, biases=np.zeros((N_layers,N)))

model.train(training_data, training_targets, n_epochs=5000, learning_rate=0.5)
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
    