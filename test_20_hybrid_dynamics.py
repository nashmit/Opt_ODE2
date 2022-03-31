from casadi import *
import numpy as np


x = SX.sym('x',1) # you can try some conditionals build around 'x' too... ( preconditions/ post conditions )
t = SX.sym('t',1)

xdot1 = 0
xdot2 = 1
xdot3 = x

t0 = 0
tf = 3

def func():
    print(xdot2)
    return xdot2

#xdot = if_else( t<1, xdot1, if_else( t<2, xdot2, xdot3 ) ) # change dynamics
xdot = if_else(t<1, xdot1, func())
print(xdot)
Function_if_else()
# don't know how to change initial condition when an event is happening ( alter current state 'x' )


ode = {'x':x, 't':t, 'ode':xdot }
time = np.insert(np.linspace(t0, tf, 20), t0, 0)

I = integrator('I','cvodes', ode, {'grid': time})

print(I(x0=1)['xf'])
