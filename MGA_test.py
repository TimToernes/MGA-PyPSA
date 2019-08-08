# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Simple test of gurobi (http://www.matthewdgilbert.com/blog/introduction-to-gurobipy.html).

import gurobipy as gurobi

m = gurobi.Model
m.setParam('OutputFlag', True)

x = m.addVar(name='x')
y = m.addVar(name='y')

m.setObjective(3*x - y, gurobi.GRB.MAXIMIZE)
m.addConstr(x + y <= 1, "c1")

m.update()
m.optimize()

for v in m.getVars():
    print('%s %s' % (v.VarName, v.X))
print('Obj: %s' % m.ObjVal)


"""   Fruit juice case           """

# Simple model with a bunch of identical ingredients. Each ingredient has a different cost.
import numpy as np

N_ingredients = 5 #Number of different ingredients.
cost = tuplelist([1.01,2,1.5,1.02,1.3,1.31]) #Per unit cost of each ingredient.

#######################################
## Initial optimization problem step ##
#######################################

model_juice = Model() #Initiate the optimization model. I.e. give it a name.
model_juice.setParam('OutputFlag', False) #How much text info you want from Gurobi. False means not so much.

ingredients = model_juice.addVars(N_ingredients) #Define the optimization variables.
model_juice.setObjective(ingredients.prod(cost), GRB.MINIMIZE) #Define the objective function. Here, it is the sum of all costs.
model_juice.addConstr(ingredients.sum() == 1) #Add constraints. Here, all indgredients must sum to one.

model_juice.update() #Update the model with new objective functions and constraints etc.
model_juice.optimize() #Solve the model.

model_solution_0 = model_juice.getVars() #Read the optimal parameter values. I.e. mix of ingredients.
model_ObjVal_0 = model_juice.ObjVal #Read the optimal objective value.

# Print some information on the screen.
print(' ')
print('Problem: ')
print(model_juice.getObjective())
print('Solution')
print(model_solution_0)
print(' ')
print('============ ')
print(' ')

##############
## MGA step ##
##############

slack = 0.05 #Set the percentage of extra cost allowed for the alternatives.

model_juice.addConstr(model_juice.getObjective() <= model_ObjVal_0*(1+slack)) # Add MGA constraint using the old objective function. Note that all original constraints are still present in the model.

#Prepare some containrs for output data.
model_solutions_all = []
model_solution_MGA = model_solution_0
index = []
for i in range(N_ingredients+1): #This stop condition is a bit arbitrary. It would be better to check if the same solution reapears.
    
    model_solutions_all.append(np.array([var.X for var in model_solution_MGA]))
    index=np.where(np.sum(np.array(model_solutions_all),axis=0))[0]
    
    print('Variables appearing in the current or past solutions: '+str(index))
    model_juice.setObjective(ingredients.sum(index), GRB.MINIMIZE) #Change to MGA objective function. This has to change every time a new solution is identified.

    # Update the model with the new objective function (and constraints).
    model_juice.update()
    model_juice.optimize()

    model_solution_MGA = model_juice.getVars() #Read the new solution.
    
    #Print the most recent MGA solution.
    print(' ')
    print('MGA Problem: ')
    print(model_juice.getObjective())
    print('Solution:')
    print(model_solution_MGA)
    
print(np.array(model_solutions_all)) #Print all the MGA solutions found in the itteration.