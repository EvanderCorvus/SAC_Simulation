import numpy as np

class Box():
    def __init__(self, center, width, height):
        self.width = width
        self.height = height
        self.center = center

    def contains(self, state):
        
        x, y = state[:,0], state[:,1]
        center_x = self.center[:,0]
        center_y = self.center[:,1]
        bool = np.logical_and(np.logical_and(x > center_x-self.width/2, x < center_x+self.width/2),
                              np.logical_and(y > center_y-self.height/2, y < center_y+self.height/2))
        return bool