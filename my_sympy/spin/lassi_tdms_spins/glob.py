from sympy import symbols
import os

s = symbols ("s", real=True, positive=True)
m = symbols ("m", real=True)

ORBINDICES = 'pqrstuvwxyz'

topdir = os.path.abspath (os.path.join (__file__, '..'))
