"""Monte Carlo Tree Search Solver"""
from utils import *
import pycuber as pc

class Node:
    
    def __init__(self, cube, model, actionInd, parent=None):

        self.c = 1
            
        self.cube = cube
        self.parent = parent
        self.model = model
        self.actionInd = actionInd
        self.isLeaf = True
		
        self.w, self.p = model.forward(get_state(cube))
        self.n = [0]*len(ACTIONS)
		
        self.l = None # loss component for parallel thing - for future use

		
    def update(self, w, ind):
        self.w = max(self.w, w)
        self.n[ind] += 1

        if self.parent:
            self.parent.update(w, ind)
                
    def simulate(self):

        if is_solved(self.cube):
            return [self.actionInd]

        if self.isLeaf:
            self.isLeaf = False
            self.children = {
                                    val:Node(self.cube.copy()(val), self.model, ind, self)
                                         for ind,val in enumerate(ACTIONS)
                                }
            if self.parent:
                self.parent.update(self.w, self.actionInd) # back + it should contain L

            return []

        choice = -1
        value = -1

        for ind,val in enumerate(ACTIONS):
            new_value = self.children[val].w + self.c * sum(self.n)**0.5 / (1+self.n[ind])
            if new_value > value:
                value = new_value
                choice = val
            #print(new_value, ind, choice)

        res = self.children[choice].simulate()
        return ([self.actionInd] + res) if res else []
                

	    
	
import time    
def solve(cube, model, timeLimit):
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
            return res

    
        
        
    




