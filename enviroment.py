import gym
import numpy as np
import torch as tr
from enviroment_utils import Box

def force(x,y,U0,type='mexican'):
    if type == 'mexican':
        r = np.sqrt(x**2+y**2)
        bool = r>0.5
        fr = -64*U0*(r**2-0.25)
        fr[bool] = 0
        # f = tr.stack([fr*x,fr*y],dim=1).to(device)
        F_x = fr*x
        F_y = fr*y
        return F_x,F_y
    
class Box2DEnv(gym.Env):
    def __init__(self, space_box, goal_box): 
        super(Box2DEnv, self).__init__()
        self.space = space_box

        self.state = None
        self.goal = goal_box

    def reset(self,batch_size=1):
        self.state = np.zeros((batch_size,5))
        self.state[:,0] = -0.5*np.ones(batch_size)
        return self.state

    def step(self, action, U0, dt):
        x, y = self.state[:,0], self.state[:,1]
        theta = self.state[:,4]
        theta = theta + action[:,0]
        
        F_x, F_y = force(x,y,U0)

        e_x = np.cos(theta)
        v_x = e_x + F_x
        x_new = x + v_x*dt

        e_y = np.sin(theta)
        v_y = e_y + F_y
        y_new = y + v_y*dt

        within_constraints_bool = self.space.contains(np.array([x_new, y_new]).T)
        
        self.state[:,0][within_constraints_bool] = x_new[within_constraints_bool]
        self.state[:,1][within_constraints_bool] = y_new[within_constraints_bool]
        self.state[:,2] = F_x
        self.state[:,3] = F_y
        self.state[:,4] = theta

    def goal_check(self):
        position = self.state[:,0:2]
        booleans = self.goal.contains(position)
        return booleans
