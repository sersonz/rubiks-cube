"""Monte Carlo Tree Search Solver"""
from utils import ACTIONS

class Node:
    #constants
    #c
   
    c=1
    def __init__(self,cube,model,actionInd,parent=None):
        self.cube = cube
        self.parent = parent
        self.model = model
        self.actionInd = actionInd
        self.isLeaf = True

        self.w,self.p = model.estimate(cube)
        self.n=[0]*len(ACTIONS)

        self.l= None


    def update(self,w,ind):
        self.w = max(self.w,w)
        self.n[ind] += 1

        if self.parent:
            self.parent.update(w,ind)

    def simulate(self):

        if self.isLeaf:
            self.isLeaf = False
            self.children = {  val:Node(cube.copy()(val), model, ind)
                                   for ind,val in enumerate(ACTIONS) }
            self.update(self.w, self.actionInd) # back +it should contain L
            return
        
        choice=0
        value=0

        for ind,val in enumerate(ACTIONS):
            new_value= self.children[val].w + c *sum(slef.n)**0.5 / (1+self.n[ind])
            if new_value > value:
                value = new_value
                choice = ind

        self.children[choice].simulate()


            






       

def solve():
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
    pass
