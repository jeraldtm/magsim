import numpy as np
import scipy as sp
import xarray as xr
from .solvers import RKSolver, SimpleSolver, RK45Solver

def getModel(*args, **kwargs):
	H, Ms, alpha, gamma, ad, fl, sigma, freq, Aex, ad_u, fl_u = args
	def Precession(y, ydot, **kwargs):
		demag = Ms*np.array([0., 0., y[2]]) #Demag for thin films = mz*Ms
		return -gamma * np.cross(y, H-demag)

	def Gilbert(y, ydot, **kwargs):
		demag = Ms*np.array([0., 0., y[2]]) #Demag for thin films = mz*Ms
		return -gamma * np.cross(y, H-demag) + alpha* np.cross(y, ydot)

	def LLGS(y, ydot, t, **kwargs):
		j = np.sin(2*np.pi*freq * t)
		demag = Ms*np.array([0., 0., y[2]]) #Demag for thin films = mz*Ms
		return -gamma * np.cross(y, H-demag) + alpha/np.linalg.norm(y) * np.cross(y, ydot) + j/np.linalg.norm(y)*ad*np.cross(y, np.cross(y, sigma)) + j*fl*np.cross(y, sigma)
	
	def LLGS_alt(y, ydot, t, **kwargs):
		j = np.sin(2*np.pi*freq * t)
		demag = Ms*np.array([0., 0., y[2]]) #Demag for thin films = mz*Ms
		return -gamma/(1+alpha**2) * (np.cross(y, H-demag) + alpha/np.linalg.norm(y) * np.cross(y, np.cross(y, H-demag))) + j/np.linalg.norm(y)*ad*np.cross(y, np.cross(y, sigma)) + j*fl*np.cross(y, sigma)

	def LLGS_exchange(y, ydot, t, **kwargs):
			j = np.sin(2*np.pi*freq * t)
			demag_l = Ms*np.array([0., 0., y[2]]) #Demag for thin films = mz*Ms
			demag_u = Ms*np.array([0., 0., y[5]])
			Heff_l = H - demag_l + Aex* y[3:] #Exchange field from upper layer
			Heff_u = H - demag_u + Aex* y[:3] #Exchange field from lower layer
			m_l = -gamma/(1+alpha**2) * (np.cross(y, Heff_l) + alpha/np.linalg.norm(y) * np.cross(y, np.cross(y, Heff_l))) + j/np.linalg.norm(y)*ad*np.cross(y, np.cross(y, sigma)) + j*fl*np.cross(y, sigma)
			m_u = -gamma/(1+alpha**2) * (np.cross(y, Heff_u) + alpha/np.linalg.norm(y) * np.cross(y, np.cross(y, Heff_u))) + j/np.linalg.norm(y)*ad_u*np.cross(y, np.cross(y, sigma)) + j*fl_u*np.cross(y, sigma)
			m = m_l.tolist() + m_u.tolist()
			return m


	return Precession, Gilbert, LLGS, LLGS_alt

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
	ds: Xarray dataset
	"""

	def __init__(self, H, Ms, alpha, gamma, ad, fl, sigma, freq, Aex, ad_u, fl_u, solver = RKSolver, **kwargs):
		self.H, self.Ms, self.alpha, self.gamma = H, Ms, alpha, gamma
		self.ad, self.fl, self.sigma, self.freq = ad, fl, sigma, freq
		self.Aex, self.ad_u, self.fl_u = Aex, ad_u, fl_u
		self.solver = solver

	def setModel(self, model, **kwargs):
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
		P, G, L, L_a = getModel(self.H, self.Ms, self.alpha, self.gamma, 
			self.ad, self.fl, self.sigma, self.freq, self.Aex, self.ad_u, self.fl_u, **kwargs)
		if model == 'Precession':
		    self.model = P
		if model == 'Gilbert':
		    self.model = G
		if model == 'LLGS':
		    self.model = L
		if model == 'LLGS_alt':
			self.model = L_a

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
		a = self.solver(self.model, y0, ydot0, **kwargs)
		for i in range(int(steps)):
		    a.step(**kwargs)
		    
		self.x = np.array([np.array(a.vars).transpose()[0][i][0] for i in range(len(np.array(a.vars).transpose()[0]))])
		self.y = np.array([np.array(a.vars).transpose()[0][i][1] for i in range(len(np.array(a.vars).transpose()[0]))])
		self.z = np.array([np.array(a.vars).transpose()[0][i][2] for i in range(len(np.array(a.vars).transpose()[0]))])
		self.t = np.array(np.array(a.vars).transpose()[2]).astype('float64')
		
		self.phi = np.arctan2(self.y, self.x).astype('float64')
		self.theta = np.arctan2(np.sqrt(self.x**2 + self.y**2), self.z)
		self.AMR = np.sin(self.phi)**2*np.sin(self.theta)**2
		self.AHE = np.cos(self.theta)
		self.PHE = np.sin(self.theta)**2*np.sin(2*self.phi)

		self.ds = xr.Dataset(data_vars={'mx': ('time', self.x), 'my': ('time', self.y), 
		    'mz': ('time', self.z), 'phi':('time', self.phi), 'theta':('time', self.theta), 'AMR':('time', self.AMR), 
		    'AHE':('time', self.AHE), 'PHE':('time', self.PHE)}, coords={'time': self.t})

		if self.solver == RK45Solver:
		    self.h = np.array(a.vars).transpose()[3].astype('float64')
		    self.diff = np.array(a.vars).transpose()[4].astype('float64')
		    self.ds = self.ds.assign(timesteps = xr.DataArray(self.h, dims = 'time'))
		    self.ds = self.ds.assign(RK45diffs = xr.DataArray(self.diff, dims = 'time'))

	def minimize(self, y0, tol, max_steps, **kwargs):
		"""
		Solves differential equation and steps through time until equilibrium state is reached where abs(yn+1 - yn) < tol or max_steps is reached
		----------
		y0 : 1D numpy array with 3 components
		    initial magnetisation

		tol : float
			tolerance for equilibrium

		max_steps: int
			max number of steps to take before stopping sim

		RKSolver **kwargs
		t0: float
		    initial time

		h: float
		    timestep
		"""
		ydot0 = -self.gamma * np.cross(y0, self.H) 
		a = self.solver(self.model, y0, ydot0, **kwargs)
		y_n = np.array(a.vars[-1][0])
		diff = np.dot(abs(y0), np.ones(3))
		step = 0

		while diff > tol and step < max_steps:
			a.step(**kwargs)
			step+=1
			y_np1 = np.array(a.vars[-1][0])
			diff = np.dot(abs(y_np1 - y_n), np.ones(3))
			y_n = y_np1

		self.x = np.array([np.array(a.vars).transpose()[0][i][0] for i in range(len(np.array(a.vars).transpose()[0]))])
		self.y = np.array([np.array(a.vars).transpose()[0][i][1] for i in range(len(np.array(a.vars).transpose()[0]))])
		self.z = np.array([np.array(a.vars).transpose()[0][i][2] for i in range(len(np.array(a.vars).transpose()[0]))])
		self.t = np.array(np.array(a.vars).transpose()[2]).astype('float64')

		self.phi = np.arctan2(self.y, self.x).astype('float64')
		self.theta = np.arctan2(np.sqrt(self.x**2 + self.y**2), self.z)
		self.AMR = np.sin(self.phi)**2*np.sin(self.theta)**2
		self.AHE = np.cos(self.theta)
		self.PHE = np.sin(self.theta)**2*np.sin(2*self.phi)
		
		self.ds = xr.Dataset(data_vars={'mx': ('time', self.x), 'my': ('time', self.y), 
		    'mz': ('time', self.z), 'phi':('time', self.phi), 'theta':('time', self.theta), 'AMR':('time', self.AMR), 
		    'AHE':('time', self.AHE), 'PHE':('time', self.PHE)}, coords={'time': self.t})

		if self.solver == RK45Solver:
		    self.h = np.array(a.vars).transpose()[3].astype('float64')
		    self.diff = np.array(a.vars).transpose()[4].astype('float64')
		    self.ds = self.ds.assign(timesteps = xr.DataArray(self.h, dims = 'time'))
		    self.ds = self.ds.assign(RK45diffs = xr.DataArray(self.diff, dims = 'time'))
