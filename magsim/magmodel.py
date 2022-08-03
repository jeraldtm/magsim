import numpy as np
import scipy as sp
import xarray as xr
import numba
from scipy.constants import *
from .solvers import RKSolver, SimpleSolver, RK45Solver

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
	Bext: external field
	Ms:
	Ku
	alpha: float
				Gilbert damping
	gamma: float
				gyrromagnetic ratio
	ad: array
			ad torque effective fields
	fl: array
			fl torque effective fields
	sigma: array
			spin polarisation
	Ifunc: function(t)
			input current function
	Jex: float
			exchange coupling strength
	solver: function
			method for integration
	xs: array of mx values
	ys: array of my values
	zs: array of mz values
	ts: array of time values
	ds: Xarray dataset
	"""

	def sinusoid(t):
		return np.cos(2*np.pi*10*t)

	def __init__(self, Bext, Ms, Ku = np.array([0., 0., 0.]), alpha = 0.01, gamma = 28.025e9*2*np.pi, ad=[0.0], fl=[0.0], sigma=np.array([0.,1.,0.]),
	 Ifunc = sinusoid, Jex=0., n = 1, T = 0, V = 1e-19, h = 1e-12, solver = RKSolver, speedup = 0, Brf = 0, **kwargs):
		self.Bext, self.Ms, self.alpha, self.gamma, self.Ku = Bext, Ms, alpha, gamma, Ku
		self.ad, self.fl, self.sigma, self.Ifunc = ad, fl, sigma, Ifunc
		self.Jex, self.n = np.array(Jex), n
		self.T, self.V = T, V
		self.solver = solver
		self.speedup = speedup
		self.h = h
		self.Btherm = [[[0., 0., 0.] for i in range(self.n)]]
		self.Brf = Brf

	def setModel(self, model, **kwargs):
		"""
		sets the differential equation to be used in the simulation
		Parameters
		----------
		model: string
		    should be one of 'Precession', 'Gilbert', 'LLGS_ex'
		alpha : float
		    Gilbert damping parameter
		gamma : float
		    gyrromagnetic ratio
		Bext: numpy array with 3 components
		    external field, e.g. np.array([0., 0., 0.])
		ad: float
		    antidamping coefficient
		fl: float
		    fieldlike coefficient
		sigma: numpy array with 3 components
		    spin polarization direction, e.g. np.array([0., 0., 0.])
		Ifunc: function
			Current function I(t) that returns I given t
		Jex: numpy array n=layers components
			Exchange coupling constants
		n: int
			number of layers for LLGS_ex
		speedup: int
			0: Standard python
			1: Numba
			2: Cython
		T: float
			Temperature for stochastic sim
		V: float
			Volume of magnetic layer for stochastic sim
		"""

		def Precession(y, **kwargs):
			Beff = self.Bext + np.array([0., 0., (2*self.Ku/self.Ms - self.Ms)*y[2]]) #Demag for thin films = mz*Ms
			return -self.gamma * np.cross(y, Beff)

		def Gilbert(y, **kwargs):
			Beff = self.Bext + np.array([0., 0., (2*self.Ku/self.Ms - self.Ms)*y[2]]) #Demag for thin films = mz*Ms
			ydot = -self.gamma * np.cross(y, Beff) 
			return -self.gamma * np.cross(y, Beff) + self.alpha* np.cross(y, ydot)

		def LLGS(y, t, **kwargs):
			# I = np.sin((2*np.pi*self.freq * t) + self.phase)
			I = self.Ifunc(t)
			Beff = self.Bext + np.array([0., 0., (2*self.Ku/self.Ms - self.Ms)*y[2]])  #Demag for thin films = mz*Ms and PMA anistropy
			ydot = -self.gamma * np.cross(y, Beff)
			return -self.gamma * np.cross(y, Beff) + self.alpha/np.linalg.norm(y) * np.cross(y, ydot)\
			 + self.gamma*I*(1/np.linalg.norm(y)*self.ad[0]*np.cross(y, np.cross(y, self.sigma)) + self.fl[0]*np.cross(y, self.sigma))

		def LLGS_alt(y, t, **kwargs):
			I = self.Ifunc(t)
			Beff = self.Bext + np.array([0., 0., (2*self.Ku/self.Ms - self.Ms)*y[2]])  #Demag for thin films = mz*Ms
			return calc_ms(self.gamma, self.alpha, y, Beff, I, self.ad, self.fl, self.sigma)

		def LLGS_ex(y, ydot, t, **kwargs):
			I = self.Ifunc(t)
			calc_funcs = [calc, calc_numba]
			return calc_funcs[self.speedup](self.n, self.Bext, self.Ku, self.Ms, np.array(y), np.array(ydot), self.Jex, self.gamma, self.alpha, I, self.ad, self.fl, self.sigma, np.array(self.Btherm[-1]), self.Brf)

		def LLGS_therm(y, t, **kwargs):
			kb = 1.38e-23
			I = self.Ifunc(t)
			Beff = self.Bext + np.array([0., 0., (2*self.Ku/self.Ms - self.Ms)*y[2]]) + self.Btherm #Demag for thin films = mz*Ms and PMA anistropy
			ydot = -self.gamma * np.cross(y, Beff)
			return -self.gamma * np.cross(y, (Beff)) + self.alpha/np.linalg.norm(y) * np.cross(y, ydot)\
			 + self.gamma*I*(1/np.linalg.norm(y)*self.ad[0]*np.cross(y, np.cross(y, self.sigma)) + self.fl[0]*np.cross(y, self.sigma))

		self.model_name = model
		if model == 'Precession':
			self.model = Precession
			return None
		if model == 'Gilbert':
			self.model = Gilbert
			return None
		if model == 'LLGS':
			self.model = LLGS
			return None
		if model == 'LLGS_alt':
			self.model = LLGS_alt
			return None
		if model == 'LLGS_ex':
			self.model = LLGS_ex
			return None
		if model == 'LLGS_therm':
			self.model = LLGS_therm
			return None
		else:
			raise Exception('Invalid model name')

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
		self.a = self.solver(self.model, y0, **kwargs)
		for i in range(int(steps)):
			if self.T != 0:
				rand_vec = np.array([np.random.normal(0., size = 3).tolist() for i in range(self.n)])
				BthermMag = (1 + self.alpha**2)*np.sqrt(2*self.alpha*k*self.T/(self.gamma *self.Ms * self.V * self.h))
				self.Btherm.append((BthermMag*rand_vec).tolist()) #Thermal Field from Belrhazi
			self.a.step(**kwargs)
		self.storeOutput()

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
		self.a = self.solver(self.model, y0, **kwargs)
		y_n = self.a.vars[-1][0]
		diff = np.dot(abs(np.array(y0[:3])), np.ones(len(y0[:3])))
		step = 0

		while diff > tol and step < max_steps:
			if self.T != 0:
				# rand_vec = np.array([[np.random.normal(), np.random.normal(), np.random.normal()] for i in range(self.n)])
				rand_vec = np.array([np.random.normal(0., size = 3).tolist() for i in range(self.n)])
				BthermMag = (1 + self.alpha**2)*np.sqrt(2*self.alpha*k*self.T/(self.gamma * self.Ms * self.V * self.h))
				self.Btherm.append((BthermMag*rand_vec).tolist()) #Thermal Field from Belrhazi #Thermal Field from Belrhazi
			self.a.step(**kwargs)
			step+=1
			y_np1 = np.array(self.a.vars[-1][0])
			diff = np.dot(abs(y_np1[:3] - y_n[:3]), np.ones(len(y_n[:3])))
			y_n = y_np1
		self.storeOutput()

	def storeOutput(self, **kwargs):
		a = self.a
		var_trans = np.array(a.vars, dtype = "object").transpose()
		self.x = [var_trans[0][i][0] for i in range(len(var_trans[0]))]
		self.y = [var_trans[0][i][1] for i in range(len(var_trans[0]))]
		self.z = [var_trans[0][i][2] for i in range(len(var_trans[0]))]
		
		self.xdot = [var_trans[1][i][0] for i in range(len(var_trans[0]))]
		self.ydot = [var_trans[1][i][1] for i in range(len(var_trans[0]))]
		self.zdot = [var_trans[1][i][2] for i in range(len(var_trans[0]))]

		self.t = var_trans[2]
		
		self.phi = np.arctan2(self.y, self.x).astype('float64')
		self.theta = np.arctan2(np.sqrt(np.array(self.x)**2 + np.array(self.y)**2), np.array(self.z))
		self.AMR = np.sin(self.phi)**2*np.sin(self.theta)**2
		self.AHE = np.cos(self.theta)
		self.PHE = np.sin(self.theta)**2*np.sin(2*self.phi)

		vars_dict = {'mx': ('time', self.x), 'my': ('time', self.y), 
		    'mz': ('time', self.z), 'phi':('time', self.phi), 'theta':('time', self.theta), 'AMR':('time', self.AMR), 
		    'AHE':('time', self.AHE), 'PHE':('time', self.PHE)}

		if self.model_name == 'LLGS_ex':
			self.xs, self.ys, self.zs = [], [], []
			self.xdots, self.ydots, self.zdots = [], [], []
			for i in range(self.n):
				x = [var_trans[0][pt][3*i] for pt in range(len(var_trans[0]))]
				y = [var_trans[0][pt][3*i + 1] for pt in range(len(var_trans[0]))]
				z = [var_trans[0][pt][3*i + 2] for pt in range(len(var_trans[0]))]
				xdot = [var_trans[1][pt][3*i] for pt in range(len(var_trans[0]))]
				ydot = [var_trans[1][pt][3*i + 1] for pt in range(len(var_trans[0]))]
				zdot = [var_trans[1][pt][3*i + 2] for pt in range(len(var_trans[0]))]
				self.xs.append(x)
				self.ys.append(y)
				self.zs.append(z)
				self.xdots.append(xdot)
				self.ydots.append(ydot)
				self.zdots.append(zdot)
				vars_dict.update({f'x{i}' : ('time', x), f'y{i}' : ('time', y), f'z{i}' : ('time', z), \
					f'xdot{i}' : ('time', xdot), f'ydot{i}' : ('time', ydot), f'zdot{i}' : ('time', zdot)})

		self.ds = xr.Dataset(data_vars=vars_dict, coords={'time': self.t})

		if self.solver == RK45Solver:
		    self.h = var_trans[3].astype('float64')
		    self.diff = var_trans[4].astype('float64')
		    self.ds = self.ds.assign(timesteps = xr.DataArray(self.h, dims = 'time'))
		    self.ds = self.ds.assign(RK45diffs = xr.DataArray(self.diff, dims = 'time'))

#Core calculation functions for speeding up with numba and cython
def calc_ms(gamma, alpha, ys, ydots, Beff, I, ad, fl, sigma, Btherm, Brf, **kwargs):
	Heff = Beff + Brf*I + Btherm

	return -gamma*(np.cross(ys, np.array(Heff))) + alpha*(np.cross(ys, ydots))\
	 + I*ad*np.cross(ys, np.cross(ys, sigma))\
	 + I*fl*np.cross(ys, sigma)

	# return gamma * (-np.cross(ys, np.array(Beff) + np.array(Btherm)) - alpha\
	# 	* np.cross(ys, np.cross(ys, Beff))\
	# 	+ I*ad*np.cross(ys, np.cross(ys, sigma))\
	# 	+ I*fl*np.cross(ys, sigma))

def calc(n, Bext, Ku, Ms, ys, ydots, Jex, gamma, alpha, I, ad, fl, sigma, Btherm, Brf, **kwargs):
	ms = np.zeros(3*n)
	ys_n = np.zeros(3*(n+2))
	for i in range(3*n):
		ys_n[i + 3] = ys[i]
	
	Jex_n = np.zeros(n+2)
	for i in range(n):
		Jex_n[i+1] = Jex[i]

	for i in range(n):
		Beff = Bext[3*i:3*(i+1)]  + (2.*Ku[3*i:3*(i+1)]/Ms)*np.array([ys[3*i], ys[3*i + 1],ys[3*i + 2]]) + Jex_n[i] * ys_n[3*i:3*(i+1)] + Jex_n[i+1] * ys_n[3*(i+2):3*(i+3)] 
		ms[3*i], ms[3*i+1], ms[3*i+2] = calc_ms(gamma, alpha, ys[3*i:3*(i+1)], ydots[3*i:3*(i+1)], Beff, I, ad[i], fl[i], sigma, Btherm[i], Brf)
	return ms

@numba.njit
def nbcross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], -a[0]*b[2] + a[2]*b[0], a[0]*b[1] - a[1]*b[0]])

@numba.njit
def nbadd(a, b, c, d):
    return np.array([a[0]+b[0]+c[0] + d[0], a[1]+b[1] + c[1] + d[1], a[2] + b[2] + c[2] + d[2]])

@numba.njit
def nbnorm(a):
	return (a[0]**2 + a[1]**2 + a[2]**2)**0.5

@numba.njit
def calc_ms_numba(gamma, alpha, ys, ydots, Beff, I, ad, fl, sigma, Btherm, Brf):
	Heff = Beff + Brf*I
	return nbadd(-gamma * nbcross(ys, nbadd(Heff, Btherm, [0, 0, 0], [0, 0, 0])), alpha * nbcross(ys, ydots), I*ad*nbcross(ys, nbcross(ys, sigma)), I*fl*nbcross(ys, sigma))
	# return gamma/(1+alpha**2) * nbadd(-nbcross(ys, nbadd(Beff, Btherm, [0, 0, 0], [0, 0, 0])), -alpha * nbcross(ys, nbcross(ys, Beff)), I*ad*nbcross(ys, nbcross(ys, sigma)), I*fl*nbcross(ys, sigma))

@numba.njit
def calc_numba(n, Bext, Ku, Ms, ys, ydots, Jex, gamma, alpha, I, ad, fl, sigma, Btherm, Brf):
	ms = np.zeros(3*n)
	ys_n = np.zeros(3*(n+2))
	for i in range(3*n):
		ys_n[i + 3] = ys[i]
	
	Jex_n = np.zeros(n+2)
	for i in range(n):
		Jex_n[i+1] = Jex[i]

	for i in range(n):
		Beff = Bext[3*i:3*(i+1)]  + (2.*Ku[3*i:3*(i+1)]/Ms)*np.array([ys[3*i], ys[3*i + 1],ys[3*i + 2]]) + Jex_n[i] * ys_n[3*i:3*(i+1)] + Jex_n[i+1] * ys_n[3*(i+2):3*(i+3)]
		ms[3*i], ms[3*i+1], ms[3*i+2] = calc_ms_numba(gamma, alpha, ys[3*i:3*(i+1)], ydots[3*i:3*(i+1)], Beff, I, ad[i], fl[i], sigma, Btherm[i], Brf)
	return ms