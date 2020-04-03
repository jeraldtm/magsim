import numpy as np
import scipy as sp

class SimpleSolver():
    def __init__(self, f, y0, h, ydot0 = 0., t0 = 0.):
        self.vars = [[y0, ydot0, t0]]
        self.f = f
        self.h = h
        
    def step(self, **kwargs):
        y, ydot, t = self.vars[-1]
        self.vars.append([y + ydot*self.h, self.f(y = y, ydot = ydot, t = t), t + self.h])
        
        
class RKSolver():
    def __init__(self, f, y0, ydot0, t0 = 0., h=1):
        self.vars = [[y0, ydot0, t0]]
        self.f = f
        self.h = h
        
    def step(self, **kwargs):
        y, ydot, t = self.vars[-1]
        k1 = self.h * self.f(y = y, ydot = ydot, t = t)
        k2 = self.h * self.f(y = (y + k1/2), ydot = ydot, t = t + self.h/2)
        k3 = self.h * self.f(y = (y + k2/2), ydot = ydot, t = t + self.h/2)
        k4 = self.h * self.f(y = (y + k3), ydot = ydot, t = t + self.h)
        y_n = y + 1/6. * (k1 + 2*k2 + 2*k3 + k4)
        ydot_n = self.f(y = y_n, ydot = ydot, t = t)
        t_n = t + self.h
        self.vars.append([y_n, ydot_n, t_n])