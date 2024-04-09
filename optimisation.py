#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import numpy as np

from nsga2.evolution import Evolution 
from nsga2.problem import Problem 

class OptimisationProblem:
    def __init__(self, surrogate_model, model = 'NN'):
        self.surrogate_model = surrogate_model
        self.predictions = None  # Initialize predictions attribute
        self.individuals = []
        self.all_predictions = []
        self.model_selection = model

    def objective_function(self, x):
        # wheights definition:
        price_biogas = 0.305 #per meter cubed in USD 
        price_sludge_TP = 1.84 #USD per kg
        price_sludge_TN = 1.19 #USD per kg 
        # Call the surrogate model's predict method
        if self.model_selection == 'DeepGP':
            x = np.array(x)
            x = x.reshape(1,-1)
            self.predictions = self.surrogate_model(x).y_mean.numpy()
            # definition of single ojective function : maximisation of the profit 
            self.objective = price_biogas*(self.predictions[0,3]+self.predictions[0,4])+price_sludge_TP*self.predictions[0,8]+price_sludge_TN*self.predictions[0,7]-(self.predictions[0,9]+self.predictions[0,10])
            #COD constraint from WHO
            if self.predictions[0,0] >= 1000:
                self.objective = self.objective - 100000 #penalise the objective function when constraint not satisfied 

            self.individuals.append(x)
            self.all_predictions.append(self.predictions)

        elif self.model_selection =='NN':
            self.predictions = self.surrogate_model.predict(x)  
            # definition of single ojective function : maximisation of the profit 
            self.objective = price_biogas*(self.predictions[3]+self.predictions[4])+price_sludge_TP*self.predictions[8]+price_sludge_TN*self.predictions[7]-(self.predictions[9]+self.predictions[10])
            if self.predictions[0] >= 1000:
                self.objective = self.objective - 100000 #penalise the objective function when constraint not satisfied

            self.individuals.append(x)
            self.all_predictions.append(self.predictions)
            
        return - (self.objective)
        

    def solve(self, bounds, num_of_individuals, num_of_generations):
        # Define the problem
        prob = Problem(num_of_variables=2, objectives =[self.objective_function], variables_range = bounds, same_range=False, expand =False) 
        evo = Evolution(prob, num_of_individuals = num_of_individuals , num_of_generations =  num_of_generations)
        func = [i.objectives for i in evo.evolve()]

        print(func, self.individuals[-1], self.all_predictions[-1])
        return (func, self.individuals[-1], self.all_predictions[-1])



