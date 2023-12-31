"""Monte Carlo Tree Search Solver"""
from utils import *
import pycuber as pc
from model import ADINet
import numpy as np

class Node:
    
    def __init__(self, cube, model, actionInd, parent=None):

        #c as 2**0.5 didnt work as well as smaller coef
        
        self.c = 0.2 
            
        self.cube = cube
        self.parent = parent
        self.model = model
        self.actionInd = actionInd
        self.isLeaf = True
		
        self.w, self.p = model.forward(get_state(cube))
        self.n = [0]*len(ACTIONS)
		

		
    def update(self, w, ind):
        self.w = max(self.w, w)
        self.n[ind] += 1

        if self.parent:
            self.parent.update(w, ind)
    
    # DFS
    def check_dfs(self):
            
        if is_solved(self.cube):
            return [self.actionInd]

        if self.isLeaf:
            return None   
        
        smallest_solution = None
        for ind,val in enumerate(ACTIONS):
            from_child = self.children[val].check()
            if from_child is not None:
                if smallest_solution is None or len(from_child) < len(smallest_solution):
                    smallest_solution = from_child
        
        if smallest_solution is not None:
            return [self.actionInd] + smallest_solution
        return None            
           
    def add_one_more_layer(self):

        if self.isLeaf:
            self.isLeaf = False
            self.children = {
                                    val:Node(self.cube.copy()(val), self.model, ind, self)
                                         for ind,val in enumerate(ACTIONS)
                                }
            return 
        
        for ind,val in enumerate(ACTIONS):
            self.children[val].add_one_more_layer
    

    def simulate(self):

        # backpropagate with naive solution if found
        if is_solved(self.cube):
            return [self.actionInd]
        
        if self.isLeaf:
            #develop the leaf
            self.isLeaf = False
            
            #instantiate children leafs
            self.children = {
                                    val:Node(self.cube.copy()(val), self.model, ind, self)
                                         for ind,val in enumerate(ACTIONS)
                                }
        
            if self.parent:
                 self.parent.update(self.w,self.actionInd) 

            return []
    
        choice = ACTIONS[0]
        value = -2

        # choose the best out of all current actions
        for ind,val in enumerate(ACTIONS):

            new_value = self.children[val].w + self.c * sum(self.n)**0.5 / (1+self.n[ind])
            if new_value > value:
                value = new_value
                choice = val
            

        res = self.children[choice].simulate()
        return ([self.actionInd] + res) if res else []
                

	    
	
import time    
def solve_naive(cube, model, timeLimit):
    """
Initialize MCTS tree with initial_state as root
while solution not found and within computational limits:
    current_state = root of the MCTS tree
    while current_state is not terminal:
        if current_state is not fully expanded:
            Expand current_state by adding one or more child states
        Use trained_network to evaluate child states
        Select next state using tree policy (balance exploration and exploitation)
        current_state = next state
    Backpropagate the result (win/loss) up the tree
return solution path from root to terminal state
    """
    endTime = time.time() + timeLimit
    root = Node(cube, model, 0)
    
    while time.time()< endTime:
        res = root.simulate()

        if res:
            return res, root

from collections import deque

def solve(cube, model, timeLimit):
    
    """
    takes the result from naive approach
    search any instantiated children for being solved and skipped

    """
    endTime = time.time() + timeLimit
    root = Node(cube, model , -1)

    while time.time() < endTime:
        res = root.simulate()

        if res:
            
            #moves all leaves to be non leaf with 1 more layer under
            root.add_one_more_layer()

            #que implementation of bfs
            que = deque([root])

            #ends because there is at least 1 solution found
            while que:
                cur = que.popleft()
                if is_solved(cur.cube):

                    #get the solution from all parents
                    solution = []
                    while cur is not None:
                        solution.append(cur.actionInd)
                        cur = cur.parent
                    return solution [::-1]
                else:

                    for i in cur.children:
                        que.append(cur.children[i])


import random

def gen(k):
    l = pc.Cube()
    for _ in range(k):
          _ = l(random.choice(ACTIONS))
    return l

time_limit = 30
trials = 10

import os

for modelName in list(os.walk('models')) [0] [-1]:

    print(modelName)
    with open('results.txt', 'a') as z:
        z.write('\t'.join(map(str , [time.time(), 'Model name' , modelName, 'Time limit',time_limit,'Tried',trials])) + '\n')

    model = ADINet()

    if torch.cuda.is_available():
        data = torch.load("models/"  + modelName) ["model"]
    else:
        data = torch.load("model/" + modelName, map_location=torch.device('cpu'))["model"]
   
    model.load_state_dict(data)
    model.eval()

    for steps in range(2, 15, 2):
       
        solved = 0
        average_length_solved = 0

        for repeats in range(trials):

            initial_cube = gen(steps)
            res =solve(initial_cube.copy(), model, time_limit)

            if res and is_solved(initial_cube(' '.join(ACTIONS[i] for i in res[1:]))):
               #double checking that the cube can be actually solved
               solved += 1
               average_length_solved += len(res)-1
            elif res:
                print('ERROR',res,' '.join(ACTIONS[i] for i in res[1: ]))

    print('Scrambles', steps,'Solved',solved,'Avg length', average_length_solved/max(1, solved))
    
    with open('results.txt', 'a') as z:
        z.write('\t'.join(map(str,[time.time(),'Scrambles',steps,'Solved',solved,'Avg length',average_length_solved/max(1, solved)]))+ '\n')
         



    
        
        
    




