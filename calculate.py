#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from pysph.solver.application import Application

import pickle
import os

# Sagemaker Mount Path
prefix='/opt/ml/'
input_path=prefix+'input/data'

output_path = os.path.join(prefix, 'output/')                       # Error
model_path = os.path.join(prefix, 'model/')                         # Result
param_path = os.path.join(prefix, 
     'input/config/hyperparameters.json')     # Parameter

channel_name='training'
training_path = os.path.join(input_path, channel_name)              # The initial state

#pickleを成功させるために必要
from pysph.solver.solver import Solver

eq_path=os.path.join(training_path, 'equation.pickle')
sol_path=os.path.join(training_path, 'solver.pickle')
pa_path=os.path.join(training_path, 'particle.pickle')

class Test(Application):
#     def create_scheme(self):
#         with open('scheme.pickle', mode='rb') as f:
#             s=pickle.load(f)      
#         return s
    def create_equations(self):
        with open(eq_path, mode='rb') as f:
            eq=pickle.load(f)
        return eq(self)       
    
    def create_solver(self):
        with open(sol_path, mode='rb') as f:
            sol=pickle.load(f)
        return sol(self)        
   
    def create_particles(self):
        with open(pa_path, mode='rb') as f:
            pa=pickle.load(f)
        return pa(self)


if __name__ == '__main__':
    app = Test(fname=model_path)
    app.run()

    