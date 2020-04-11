import numpy as np
import scipy as sp
import xarray as xr
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
	H: external field
	gamma: gyrromagnetic ratio
	alpha: Gilbert damping
	xs: array of mx values
	ys: array of my values
	zs: array of mz values
	ts: array of time values
	ds: Xarray dataset
	"""

	def __init__(self, H, Ms, Ku = 0.0, alpha = 0.1, gamma = 1.76e11, ad=0.0, fl=0.0, sigma=np.array([0.,1.,0.])
		, freq = 10., phase = 0.0, Aex=0., ad_u=0., fl_u=0., n = 2, solver = RKSolver, **kwargs):
		self.H, self.Ms, self.alpha, self.gamma, self.Ku = H, Ms, alpha, gamma, Ku
		self.ad, self.fl, self.sigma, self.freq, self.phase = ad, fl, sigma, freq, phase
		self.Aex, self.ad_u, self.fl_u, self.n = Aex, ad_u, fl_u, n
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

		def Precession(y, **kwargs):
			Heff = H + np.array([0., 0., 2*self.Ku/self.Ms - self.Ms*y[2]]) #Demag for thin films = mz*Ms
			return -self.gamma * np.cross(y, self.H-demag)

		def Gilbert(y, **kwargs):
			Heff = self.H + np.array([0., 0., 2*self.Ku/self.Ms - self.Ms*y[2]]) #Demag for thin films = mz*Ms
			ydot = -self.gamma * np.cross(y, Heff) 
			return -self.gamma * np.cross(y, Heff) + self.alpha* np.cross(y, ydot)

		def LLGS(y, t, **kwargs):
			j = np.sin((2*np.pi*self.freq * t) + self.phase)
			Heff = self.H + np.array([0., 0., 2*self.Ku/self.Ms - self.Ms*y[2]])  #Demag for thin films = mz*Ms and PMA anistropy
			ydot = -self.gamma * np.cross(y, Heff)
			return -self.gamma * np.cross(y, Heff) + self.alpha/np.linalg.norm(y) * np.cross(y, ydot)\
			 + j/np.linalg.norm(y)*self.ad*np.cross(y, np.cross(y, self.sigma)) + j*self.fl*np.cross(y, self.sigma)

		def LLGS_alt(y, t, **kwargs):
			j = np.sin((2*np.pi*self.freq * t) + self.phase)
			Heff = self.H + np.array([0., 0., 2*self.Ku/self.Ms - self.Ms*y[2]])  #Demag for thin films = mz*Ms
			return -self.gamma/(1+self.alpha**2) * (np.cross(y, Heff) + self.alpha/np.linalg.norm(y) * np.cross(y, np.cross(y, Heff)))\
			 + j/np.linalg.norm(y)*self.ad*np.cross(y, np.cross(y, self.sigma)) + j*self.fl*np.cross(y, self.sigma)

		def LLGS_ex(y, t, **kwargs):
			ys, Heffs, ms = [], [], []
			j = np.sin((2*np.pi*self.freq * t) + self.phase)
			for i in range(self.n):
				ys.append(y[3*i:3*(i+1)])

			for i in range(self.n):
				if i == 0:
					Heffs.append(self.H + np.array([0., 0., 2*self.Ku/self.Ms - self.Ms*ys[0][2]]) + self.Aex * np.array(ys[i+1]))
				if i == (self.n-1):
					Heffs.append(self.H + np.array([0., 0., 2*self.Ku/self.Ms - self.Ms*ys[i][2]]) + self.Aex * np.array(ys[i-1]))
				if i != 0 and i != (self.n-1):
					Heffs.append(self.H + np.array([0., 0., 2*self.Ku/self.Ms - self.Ms*ys[i][2]]) + self.Aex * (np.array(ys[i-1]) + np.array(ys[i+1])))

			for i in range(self.n):
				if i == 0:
					ms.append(-self.gamma/(1+self.alpha**2) * (np.cross(ys[0], Heffs[0]) + self.alpha/np.linalg.norm(ys[0])\
					 * np.cross(ys[0], np.cross(ys[0], Heffs[0]))) + j/np.linalg.norm(ys[0])*self.ad*np.cross(ys[0], np.cross(ys[0], self.sigma))\
					  + j*self.fl*np.cross(ys[0], self.sigma))
				
				if i != 0:
					ms.append(-self.gamma/(1+self.alpha**2) * (np.cross(ys[i], Heffs[i]) + self.alpha/np.linalg.norm(ys[i])\
					 * np.cross(ys[i], np.cross(ys[i], Heffs[i]))) + j/np.linalg.norm(ys[i])*self.ad_u*np.cross(ys[i], np.cross(ys[i], self.sigma))\
					  + j*self.fl_u*np.cross(ys[i], self.sigma))
				
			m = []
			for i in range(self.n):
				m += ms[i].tolist()
			
			return np.array(m)

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
		y_n = np.array(self.a.vars[-1][0])
		diff = np.dot(abs(y0), np.ones(len(y0)))
		step = 0

		while diff > tol and step < max_steps:
			self.a.step(**kwargs)
			step+=1
			y_np1 = np.array(self.a.vars[-1][0])
			diff = np.dot(abs(y_np1 - y_n), np.ones(len(y_n)))
			y_n = y_np1

		self.storeOutput()

	def storeOutput(self, **kwargs):
		a = self.a
		self.x = np.array([np.array(a.vars).transpose()[0][i][0] for i in range(len(np.array(a.vars).transpose()[0]))])
		self.y = np.array([np.array(a.vars).transpose()[0][i][1] for i in range(len(np.array(a.vars).transpose()[0]))])
		self.z = np.array([np.array(a.vars).transpose()[0][i][2] for i in range(len(np.array(a.vars).transpose()[0]))])
		self.t = np.array(np.array(a.vars).transpose()[2]).astype('float64')
		
		self.phi = np.arctan2(self.y, self.x).astype('float64')
		self.theta = np.arctan2(np.sqrt(self.x**2 + self.y**2), self.z)
		self.AMR = np.sin(self.phi)**2*np.sin(self.theta)**2
		self.AHE = np.cos(self.theta)
		self.PHE = np.sin(self.theta)**2*np.sin(2*self.phi)

		vars_dict = {'mx': ('time', self.x), 'my': ('time', self.y), 
		    'mz': ('time', self.z), 'phi':('time', self.phi), 'theta':('time', self.theta), 'AMR':('time', self.AMR), 
		    'AHE':('time', self.AHE), 'PHE':('time', self.PHE)}

		if self.model_name == 'LLGS_ex':
			self.xs, self.ys, self.zs = [], [], []
			for i in range(self.n):
				x = np.array([np.array(a.vars).transpose()[0][pt][3*i] for pt in range(len(np.array(a.vars).transpose()[0]))])
				y = np.array([np.array(a.vars).transpose()[0][pt][3*i + 1] for pt in range(len(np.array(a.vars).transpose()[0]))])
				z = np.array([np.array(a.vars).transpose()[0][pt][3*i + 2] for pt in range(len(np.array(a.vars).transpose()[0]))])
				self.xs.append(x)
				self.ys.append(y)
				self.zs.append(z)
				vars_dict.update({f'x{i}' : ('time', x), f'y{i}' : ('time', y), f'z{i}' : ('time', z)})

		self.ds = xr.Dataset(data_vars=vars_dict, coords={'time': self.t})

		if self.solver == RK45Solver:
		    self.h = np.array(a.vars).transpose()[3].astype('float64')
		    self.diff = np.array(a.vars).transpose()[4].astype('float64')
		    self.ds = self.ds.assign(timesteps = xr.DataArray(self.h, dims = 'time'))
		    self.ds = self.ds.assign(RK45diffs = xr.DataArray(self.diff, dims = 'time'))
