from casadi import *
import numpy as np

x = SX.sym('x',1)

xdot = DM(1)
#tf = DM([3])
tf = SX.sym('tf',1)
t0 = DM([0])

# by taking derivative wrt time for Mayer(tf) which is "t"
L = 1
#L = x
dae = { 'x':x, 'ode':xdot, 'quad':L }
I1 = integrator('I1', 'idas',dae , {'t0':0, 'tf':3 })

xdot2 = xdot * (tf-t0)
L2 = 1*(tf-t0) #3*1 -->> must normalize 'Lagrangian' too!
#L2 = x*(tf-t0) #3*x -->> must normalize 'Lagrangian' too!
p = tf # we have one parameter now...
dae2 = {'x':x, 'p':p, 'ode':xdot2, 'quad':L2 }
I2 = integrator('I2', 'idas', dae2, {'t0':0 , 'tf':1 })

res1 = I1(x0=1)
print(res1)

res2 = I2(x0=1,p=3)
print(res2)

print('I1 -> [t0,tf] = [0,3]')
print('I2 -> [t0,tf] = [0,1] ( normalized )')

print( "res1['qf'] / res2['qf'] == ",
       res1['qf'] / res2['qf'],
       "the result of the quadrature is also normalized !" )