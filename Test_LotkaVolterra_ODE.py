from PlottingSystem import OCP_Plot

import matplotlib.pyplot as plt

from DAE import *


from casadi import *
import numpy as np

from enum import Enum
from copy import deepcopy
from typing import Union

from homework2 import *

# idas ODE

# sumsqr( LotkaVolterra_ODE.GetX() ) + sumsqr( LotkaVolterra_ODE.GetW() )
LotkaVolterra_ODE = ClassDefineLotka_Volterra("LotkaVolterra")
ocp = OCP( "Optimize LotkaVolterra" ). \
    AddDAE( LotkaVolterra_ODE ). \
    AddLagrangeCostFunction( L = sumsqr( LotkaVolterra_ODE.GetX() ) + sumsqr( LotkaVolterra_ODE.GetW() ) ). \
    SetStartTime( t0 = Config_LotkaVolterra['t0'] ). \
    SetEndTime( tf = Config_LotkaVolterra['tf'] ). \
    SetX_0( x0 = Config_LotkaVolterra['Sn'], mask=Config_LotkaVolterra['S0_mask'] ). \
    SetX_f( xf = Config_LotkaVolterra['Xf'], mask=Config_LotkaVolterra['Xf_mask'] ). \
    SetLBW(lbw=Config_LotkaVolterra['lbw'] ). \
    SetUBW(ubw=Config_LotkaVolterra['ubw'] ). \
    SetLBQ(lbq=Config_LotkaVolterra['lbq']). \
    SetUBQ(ubq=Config_LotkaVolterra['ubq']). \
    SetNumberShootingNodes( Number = Config_LotkaVolterra['NumberShootingNodes'] ). \
    SetSolver( Solver = OptSolver.nlp )
#    Build()
ocp.Build( Config_LotkaVolterra )
