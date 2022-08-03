import numpy as np
import scipy as sp

class SimpleSolver():
    def __init__(self, f, y0, ydot0 = 0., t0 = 0.,  h=1, **kwargs):
        self.vars = [[y0, ydot0, t0]]
        self.f = f
        self.h = h
        
    def step(self, **kwargs):
        y, ydot, t, h = self.vars[-1]
        self.vars.append([y + ydot*self.h, self.f(y = y, ydot = ydot, t = t, **kwargs), t + self.h])
        
        
class RKSolver():
    def __init__(self, f, y0, ydot0 = 0., t0 = 0., h=1, **kwargs):
        self.vars = [[y0, ydot0, t0]]
        self.f = f
        self.h = h
        
    def step(self, **kwargs):
        y, ydot, t= self.vars[-1]
        k1 = self.f(y = y, ydot = ydot, t = t, **kwargs)
        k2 = self.f(y = (y + self.h*k1/2), ydot = ydot, t = t + self.h/2, **kwargs)
        k3 = self.f(y = (y + self.h*k2/2), ydot = ydot, t = t + self.h/2, **kwargs)
        k4 = self.f(y = (y + self.h*k3), ydot = ydot, t = t + self.h, **kwargs)
        y_n = y + 1/6. * self.h*(k1 + 2*k2 + 2*k3 + k4)
        ydot_n = self.f(y = y_n, ydot = ydot, t = t, **kwargs)
        t_n = t + self.h
        self.vars.append([y_n.tolist(), ydot_n, t_n])

class RK45Solver():
	def __init__(self, f, y0, ydot0 = 0., t0 = 0., h=1, hlim = [1e-14, 1e-10], eps = 0.35, **kwargs):
	    self.vars = [[y0, ydot0, t0, h, 0]]
	    self.f = f
	    self.h = h
	    self.eps = eps
	    self.hlim = hlim
        
	def step(self, **kwargs):
		a = [1/4., [1/4.]]
		b = [3/8., [3/32., 9/32.]]
		c = [12/13., [1932/2197., -7200/2197., 7296/2197.]]
		d = [1., [439/216., -8., 3680/513., -845/4104.]]
		e = [1/2., [-8/27., 2., -3544/2565., 1859/4104., -11/40.]]
		f = [16/135., 0., 6656/12825., 28561/56430., -9/50., 2/55.]
		g = [25/216., 0., 1408/2565., 2197/4104., -1/5., 0.]
		y, ydot, t , h, diffs = self.vars[-1]

		k1 = self.f(y = y, ydot = ydot, t = t, **kwargs)
		t1 = t + a[0]*h
		y1 = y + h*a[1][0] * k1

		k2 = self.f(y = y1, ydot = ydot, t = t1, **kwargs)
		t2 = t + b[0]*h
		y2 = y + h*(b[1][0]*k1 + b[1][1]*k2)

		k3 = self.f(y = y2, ydot = ydot, t = t2, **kwargs)
		t3 = t + c[0]*h
		y3 = y + h*(c[1][0]*k1 + c[1][1]*k2 + c[1][2]*k3)

		k4 = self.f(y = y3, ydot = ydot, t = t3, **kwargs)
		t4 = t + d[0]*h
		y4 = y + h*(d[1][0]*k1 + d[1][1]*k2 + d[1][2]*k3 + d[1][3]*k4)

		k5 = self.f(y = y4, ydot = ydot, t=t4, **kwargs)
		t5 = t + e[0]*h
		y5 = y + h*(e[1][0]*k1 + e[1][1]*k2 + e[1][2]*k3 + e[1][3]*k4 + e[1][4]*k5)

		k6 = self.f(y=y5, ydot = ydot, t=t5, **kwargs)

		y_n4 = y + self.h*(f[0]*k1 + f[1]*k2 + f[2]*k3 + f[3]*k4 + f[4]*k5 +f[5]*k6)
		y_n5 = y + self.h*(g[0]*k1 + g[1]*k2 + g[2]*k3 + g[3]*k4 + g[4]*k5 +g[5]*k6)

		y_n = (y_n4 + y_n5)/2.
		ydot_n = self.f(y = y_n, ydot = ydot, t = t, **kwargs)
		t_n = t + self.h

		#variable timestep
		diff = abs(y_n4[0] - y_n5[0]) + abs(y_n4[1] - y_n5[1]) + abs(y_n4[2] - y_n4[2])
		if diff !=0:
			hnew = self.h* (self.eps/(2*diff)**(0.25))
		elif diff == 0:
			hnew = self.hlim[1]
		if (hnew <= self.hlim[1]):
			if (hnew >= self.hlim[0]):
				self.h = hnew
		self.vars.append([y_n, ydot_n, t_n, self.h, diff])