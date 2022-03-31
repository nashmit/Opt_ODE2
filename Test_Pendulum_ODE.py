from casadi import *
import numpy as np

from DAE import ClassDefinePendulum, Config_Pendulum
from homework2 import OCP, OptSolver

Pendulum_ODE = ClassDefinePendulum('Pendulum_ODE')
#tf = SX.sym('tf',1)
#Mayer_exp = tf
#AddMayerCostFunction( M = Mayer_exp ). \
#SetEndTime( tf = tf ). \
ocp = OCP('Pendulum'). \
    SetDAE( Pendulum_ODE ). \
    AddLagrangeCostFunction( L = 0 ). \
    SetStartTime( t0 = Config_Pendulum['t0'] ). \
    SetX_0( x0 = Config_Pendulum['Sn'], mask=Config_Pendulum['S0_mask'] ). \
    SetX_f( xf = Config_Pendulum['Xf'], mask=Config_Pendulum['Xf_mask'] ). \
    SetLBW(lbw=Config_Pendulum['lbw'] ). \
    SetUBW(ubw=Config_Pendulum['ubw'] ). \
    SetLBQ(lbq=Config_Pendulum['lbq'] ). \
    SetUBQ(ubq=Config_Pendulum['ubq'] ). \
    SetNumberShootingNodes( Number = Config_Pendulum['NumberShootingNodes'] ). \
    SetSolver( Solver = OptSolver.nlp ). \
    SetMaxIterationNumber( max_iter = 12 )
    #    Build()
ocp.Build( Config_Pendulum )

#SetEndTime( tf = Config_Pendulum['tf'] ). \

