from casadi import *
#from casadi.tools.graph.graph import *
import numpy as np

x = MX.sym('x',1)
t = MX.sym('t',1)
xdot = x
L= x
# L = 1 for computing the time.
ode = {'x':x, 't':t, 'ode':xdot, 'quad':L}
I = integrator('I', 'cvodes', ode, {'t0': 0, 'tf': 4})
print(I)
y = MX.sym('y',1)
print(I(x0=x))
print(I(x0=1))