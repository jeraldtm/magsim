import numpy as np
import scipy as sp
import xarray as xr
import numba
from .solvers import RKSolver, SimpleSolver, RK45Solver
from .cythonFunctions import cythonCross, cythonAdd, cythonMult, cythonNorm

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

	def __init__(self, Bext, Ms, Ku = 0.0, alpha = 0.1, gamma = 1.76e11, ad=[0.0], fl=[0.0], sigma=np.array([0.,1.,0.])
		, freq = 10., phase = 0.0, Jex=0., n = 2, solver = RKSolver, speedup = 0, **kwargs):
		self.Bext, self.Ms, self.alpha, self.gamma, self.Ku = Bext, Ms, alpha, gamma, Ku
		self.ad, self.fl, self.sigma, self.freq, self.phase = ad, fl, sigma, freq, phase
		self.Jex, self.n = Jex, n
		self.solver = solver
		self.speedup = speedup

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
		Bext: numpy array with 3 components
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
			Beff = self.Bext + np.array([0., 0., (2*self.Ku/self.Ms - self.Ms)*y[2]]) #Demag for thin films = mz*Ms
			return -self.gamma * np.cross(y, Beff)

		def Gilbert(y, **kwargs):
			Beff = self.Bext + np.array([0., 0., (2*self.Ku/self.Ms - self.Ms)*y[2]]) #Demag for thin films = mz*Ms
			ydot = -self.gamma * np.cross(y, Beff) 
			return -self.gamma * np.cross(y, Beff) + self.alpha* np.cross(y, ydot)

		def LLGS(y, t, **kwargs):
			I = np.sin((2*np.pi*self.freq * t) + self.phase)
			Beff = self.Bext + np.array([0., 0., (2*self.Ku/self.Ms - self.Ms)*y[2]])  #Demag for thin films = mz*Ms and PMA anistropy
			ydot = -self.gamma * np.cross(y, Beff)
			return -self.gamma * np.cross(y, Beff) + self.alpha/np.linalg.norm(y) * np.cross(y, ydot)\
			 + self.gamma*I*(1/np.linalg.norm(y)*self.ad[0]*np.cross(y, np.cross(y, self.sigma)) + self.fl[0]*np.cross(y, self.sigma))

		def LLGS_alt(y, t, **kwargs):
			I = np.sin((2*np.pi*self.freq * t) + self.phase)
			Beff = self.Bext + np.array([0., 0., (2*self.Ku/self.Ms - self.Ms)*y[2]])  #Demag for thin films = mz*Ms
			return calc_ms(self.gamma, self.alpha, y, Beff, I, self.ad, self.fl, self.sigma)

		def LLGS_ex(y, t, **kwargs):
			ys, Beffs, ms = [], [], []
			I = np.sin((2*np.pi*self.freq * t) + self.phase)
			for i in range(self.n):
				ys.append(y[3*i:3*(i+1)])

			calc_beffs_funcs = [calc_beffs, calc_beffs_numba, calc_beffs_cython]
			calc_ms_funcs = [calc_ms, calc_ms_numba, calc_ms_cython]

			for i in range(self.n):
				if i == 0:
					if self.n == 1:
						Beffs.append(calc_beffs_funcs[self.speedup](self.Bext, self.Ku, self.Ms, ys[0][2], self.Jex, np.zeros(3)))
					else:
						Beffs.append(calc_beffs_funcs[self.speedup](self.Bext, self.Ku, self.Ms, ys[0][2], self.Jex, np.array(ys[1])))

				if i != 0 and i != (self.n-1):
					Beffs.append(calc_beffs_funcs[self.speedup](self.Bext, self.Ku, self.Ms, ys[i][2], self.Jex, (np.array(ys[i-1]) + np.array(ys[i+1]))))

				if i == (self.n-1):
					Beffs.append(calc_beffs_funcs[self.speedup](self.Bext, self.Ku, self.Ms, ys[i][2], self.Jex, (np.array(ys[i-1]))))

			for i in range(self.n):
				if i == 0:
					ms.append(calc_ms_funcs[self.speedup](self.gamma, self.alpha, ys[0], Beffs[0], I, self.ad[0], self.fl[0], self.sigma))
				if i != 0:
					ms.append(calc_ms_funcs[self.speedup](self.gamma, self.alpha, ys[i], Beffs[i], I, self.ad[i], self.fl[i], self.sigma))

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
		diff = np.dot(abs(y0[:3]), np.ones(len(y0[:3])))
		step = 0

		while diff > tol and step < max_steps:
			self.a.step(**kwargs)
			step+=1
			y_np1 = np.array(self.a.vars[-1][0])
			diff = np.dot(abs(y_np1[:3] - y_n[:3]), np.ones(len(y_n[:3])))
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

#Multi use Core functions
def calc_ms(gamma, alpha, ys, Beff, I, ad, fl, sigma):
    return -gamma/(1+alpha**2) * (np.cross(ys, Beff) + alpha/np.linalg.norm(ys)\
                     * np.cross(ys, np.cross(ys, Beff)))\
                      + gamma/(1+alpha**2)*I*(1/np.linalg.norm(ys)*ad*np.cross(ys, np.cross(ys, sigma))\
                      + fl*np.cross(ys, sigma))

def calc_beffs(Bext, Ku, Ms, ysz, Jex, ys_nn):
	return Bext + np.array([0., 0., (2*Ku/Ms - Ms)*ysz]) + Jex * Ms * ys_nn

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
def calc_ms_numba(gamma, alpha, ys, Beff, I, ad, fl, sigma):
    return gamma/(1+alpha**2) * nbadd(-nbcross(ys, Beff), -alpha/nbnorm(ys)\
                     * nbcross(ys, nbcross(ys, Beff)), I*ad/nbnorm(ys)*nbcross(ys, nbcross(ys, sigma)), fl*nbcross(ys, sigma))

@numba.njit
def calc_beffs_numba(Bext, Ku, Ms, ysz, Jex, ys_nn):
	return Bext + np.array([0., 0., (2*Ku/Ms - Ms)*ysz]) + Jex * Ms * ys_nn


def calc_ms_cython(gamma, alpha, ys, Beff, I, ad, fl, sigma):
    return cythonMult(cythonAdd(-cythonCross(ys, Beff), -cythonMult(cythonCross(ys, cythonCross(ys, Beff)), alpha/cythonNorm(ys)), \
    	cythonMult(cythonCross(ys, cythonCross(ys, sigma)), I*ad/cythonNorm(ys)), cythonMult(cythonCross(ys, sigma), I*fl)),\
    	 gamma/(1+alpha**2))

def calc_beffs_cython(Bext, Ku, Ms, ysz, Jex, ys_nn):
	return cythonAdd(Bext, np.array([0., 0., (2*Ku/Ms - Ms)*ysz]), cythonMult(ys_nn, Jex*Ms), np.zeros(3))
