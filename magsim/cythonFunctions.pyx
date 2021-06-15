import time
import numpy
cimport numpy

def cythonCross(numpy.ndarray[numpy.float_t, ndim = 1] a, numpy.ndarray[numpy.float_t, ndim = 1] b):
	cdef float a0, a1, a2, b0, b1, b2
	cdef numpy.ndarray[numpy.float_t, ndim = 1] out = numpy.ones(3)

	a0, a1, a2, b0, b1, b2 = a[0], a[1], a[2], b[0], b[1], b[2]
	out[0] = a[1]*b[2] - a[2]*b[1]
	out[1] = -a[0]*b[2] + a[2]*b[0]
	out[2] = a[0]*b[1] - a[1]*b[0]
	return out

def cythonAdd(numpy.ndarray[numpy.float_t, ndim = 1] a, numpy.ndarray[numpy.float_t, ndim = 1] b, numpy.ndarray[numpy.float_t, ndim = 1] c, numpy.ndarray[numpy.float_t, ndim = 1] d):
	cdef float a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2
	cdef numpy.ndarray[numpy.float_t, ndim = 1] out = numpy.ones(3)

	a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2 = a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]
	out[0] = a[0] + b[0] + c[0] + d[0]
	out[1] = a[1] + b[1] + c[1] + d[1]
	out[2] = a[2] + b[2] + c[2] + d[2]
	return out

def cythonMult(numpy.ndarray[numpy.float_t, ndim = 1] a, float b):
	cdef float a0, a1, a2
	cdef numpy.ndarray[numpy.float_t, ndim = 1] out = numpy.ones(3)
	a0, a2, a2 = a[0], a[1], a[2]
	out[0] = a[0]*b
	out[1] = a[1]*b
	out[2] = a[2]*b
	return out

def cythonNorm(numpy.ndarray[numpy.float_t, ndim = 1] a):
	cdef float a0, a1, a2
	a0, a1, a2 = a[0], a[1], a[2]
	cdef float amp = (a[0]**2 + a[1]**2 + a[2]**2)**0.5
	return amp