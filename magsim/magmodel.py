import numpy as np
import scipy as sp
from .solvers import RKSolver, SimpleSolver

def getModel(*args, **kwargs):
    alpha, gamma, H, ad, fl, sigma, freq = args
    def Precession(y, ydot, H = H, gamma = gamma, **kwargs):
            return -gamma * np.cross(y, H)

    def Gilbert(y, ydot, H = H, alpha = alpha, gamma = gamma, **kwargs):
            return -gamma * np.cross(y, H) + alpha/np.linalg.norm(y) * np.cross(y, ydot)

    def LLGS(y, ydot, t, H = H, alpha = alpha, gamma = gamma, ad = ad, fl = fl, sigma = sigma, freq = freq):
            j = np.sin(2*np.pi*freq * t)
            return -gamma * np.cross(y, H) + alpha/np.linalg.norm(y) * np.cross(y, ydot) + j*ad*np.cross(y, np.cross(y, sigma)) + j*fl*np.cross(y, sigma)
    return Precession, Gilbert, LLGS

class MagModel():
    """
    Class to contain magnet dynamics by solving the LLGS equations

    Parameters
    ----------
    method : str
        Should be 'RK' or 'Euler', to specify differential equation solver

    Attributes
    ----------
    f : differential equation governing time evolution of m.
    H: external field
    gamma: gyrromagnetic ratio
    alpha: Gilbert damping
    xs: array of mx values
    ys: array of my values
    zs: array of mz values
    ts: array of time values
    """

    def setModel(self, model, alpha, gamma, H, ad, fl, sigma, freq, **kwargs):
        """
        sets the differential equation to be used in the simulation
        Parameters
        ----------
        model: string
            should be 'Precession', 'Gilbert', or 'LLGS'
        alpha : float
            Gilbert damping parameter
        gamma : float
            gyrromagnetic ratio
        H: numpy array with 3 components
            external field, e.g. np.array([0., 0., 0.])
        ad: float
            antidamping coefficient
        fl: float
            fieldlike coefficient
        sigma: numpy array with 3 components
            spin polarization direction, e.g. np.array([0., 0., 0.])
        freq: float
            current injection frequency for LLGS model
        """
        self.alpha, self.gamma, self.H, self.ad, self.fl, self.sigma, self.freq = alpha, gamma, H, ad, fl, sigma, freq
        P, G, L = getModel(alpha, gamma, H, ad, fl, sigma, freq)
        if model == 'Precession':
            self.model = P
        if model == 'Gilbert':
            self.model = G
        if model == 'LLGS':
            self.model = L
    
    def runModel(self, steps, y0, **kwargs):
        """
        Solves differential equation and outputs mx, my and mz values
        ----------
        steps: int
            number of time steps
        y0 : 1D numpy array with 3 components
            initial magnetisation
        
        RKSolver **kwargs
        t0: float
            initial time

        h: float
            timestep
        """
        ydot0 = -self.gamma * np.cross(y0, self.H) 
        a = RKSolver(self.model, y0, ydot0, **kwargs)
        for i in range(int(steps)):
            a.step()
            
        self.xs = [np.array(a.vars).transpose()[0][i][0] for i in range(len(np.array(a.vars).transpose()[0]))]
        self.ys = [np.array(a.vars).transpose()[0][i][1] for i in range(len(np.array(a.vars).transpose()[0]))]
        self.zs = [np.array(a.vars).transpose()[0][i][2] for i in range(len(np.array(a.vars).transpose()[0]))]
        self.ts = np.array(a.vars).transpose()[2]