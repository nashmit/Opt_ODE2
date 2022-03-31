from casadi import *
import numpy as np

def DefineOneShootingNodForVanDerPol(t0, tf, NrEvaluationPoints = 40):
    x0 = SX.sym('x0')
    x1 = SX.sym('x1')
    x = vertcat(x0, x1)
    c = SX.sym('c')
    k = SX.sym('k')
    p = vertcat(c, k)
    xdot = vertcat(
        x1,
        c * ( 1 - x0 * x0 ) * x1 - k * x0
    )

    ode = { 'x':x, 'p':p, 'ode':xdot }

    I = integrator('I', 'cvodes', ode, {'t0': t0, 'tf': tf})

    time = np.insert(np.linspace(t0, tf, NrEvaluationPoints), t0, 0)
    I_plot = integrator('I_plot', 'cvodes', ode, {'grid': time})

    # returns [integrator][Nr equations][Nr parameters]
    return I, 2, 2, I_plot

t0 = 0 # MX.sym('t0',1)
tf = 2 #MX.sym('tf',1)

DefineOneShootingNodForVanDerPol(t0,tf)
