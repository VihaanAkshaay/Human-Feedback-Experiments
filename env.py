import numpy
import random

class RandomNumberEnvironment():
    def __init__(self,number):
        self.number = number
        self.state = 0
        self.done = False

    def reset(self):

        self.state = 0
        self.done = False
        return self.state
    
    def state_boundaries(self):
        if self.state == -self.number:
            return -1
    
    def step(self,action):

        if action == 0:
            self.state = self.state
        elif action == 1:
            self.state = self.state + 1
        elif action == -1:
            self.state = self.state - 1
        else:
            print("Your action is", action)
            raise ValueError("Action must be 0, 1 or -1")
        
        if self.state == self.number:
            self.done = True
        
        return self.state, self.reward(), self.done, None
    
    def reward(self):
        if self.done:
            return 1
        else:
            return 0
        
    def render(self):   
        print(self.state)