from casadi import *
#from casadi.tools.graph.graph import *
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# https://mathinsight.org/two_dimensional_autonomous_differential_equation_problems

x = MX.sym('x',2)
t = MX.sym('t',1)
#xdot = vertcat( -x[0]-x[1]+1, x[0]-x[1]+1 )
xdot = vertcat( 1,1)
#L=xdot
L = x
ode = {'x':x, 't':t, 'ode':xdot, 'quad':L}
#I = integrator('I', 'cvodes', ode, {'grid':[0, 1, 2, 3,4 ,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}) #{'t0': 0, 'tf': 2})
I = integrator('I', 'cvodes', ode, {'grid':[0, 0, 0.1, 0.2, 0.3, 0.4, 0.5]}) #{'t0': 0, 'tf': 2})
#I = integrator('I', 'cvodes', ode, {'grid':[0, 0, 1, 2, 3, 4, 5,6,7,8,9]}) #{'t0': 0, 'tf': 2})

print(I)

print(I(x0=x))
print(I(x0=[1,1]))

res = I(x0=[1,1])['qf']
plt.plot(  res[0,:].full()[0], 'x')
#plt.plot(2,3)
plt.axis('equal')
plt.show()