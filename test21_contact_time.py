from casadi import *
import numpy as np

from DAE import ClassDefineBall, Config_Ball
from homework2 import OCP, OptSolver

Ball_ODE = ClassDefineBall('Ball_ODE')
tf = SX.sym('tf',1)
Mayer_exp = tf
ocp = OCP('Contact time').\
    AddDAE( Ball_ODE ). \
    AddLagrangeCostFunction( L = 0 ). \
    AddMayerCostFunction( M = Mayer_exp ). \
    SetStartTime( t0 = Config_Ball['t0'] ). \
    SetEndTime( tf = tf ). \
    SetX_0( x0 = Config_Ball['Sn'], mask=Config_Ball['S0_mask'] ). \
    SetX_f( xf = Config_Ball['Xf'], mask=Config_Ball['Xf_mask'] ). \
    SetLBW(lbw=Config_Ball['lbw'] ). \
    SetUBW(ubw=Config_Ball['ubw'] ). \
    SetLBQ(lbq=Config_Ball['lbq'] ). \
    SetUBQ(ubq=Config_Ball['ubq'] ). \
    SetNumberShootingNodes( Number = Config_Ball['NumberShootingNodes'] ). \
    SetSolver( Solver = OptSolver.nlp )
#    Build()
ocp.Build( Config_Ball )
