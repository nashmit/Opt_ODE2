from PlottingSystem import OCP_Plot

import matplotlib.pyplot as plt

from DAE import *


from casadi import *
import numpy as np

from enum import Enum
from copy import deepcopy
from typing import Union

from homework2 import *

VanDerPol_ODE = ClassDefineOneShootingNodForVanDerPol("VanDerPol")
sumsqr( VanDerPol_ODE.GetW() ) + sumsqr( VanDerPol_ODE.GetX() )
ocp = OCP( "Optimize VanDerPol" ). \
    AddDAE( VanDerPol_ODE ). \
    AddLagrangeCostFunction( L = 0 ). \
    SetStartTime( t0 = Config_VanDerPol['t0'] ). \
    SetEndTime( tf = Config_VanDerPol['tf'] ). \
    SetX_0( x0 = Config_VanDerPol['Sn'], mask=Config_VanDerPol['S0_mask'] ). \
    SetX_f( xf = Config_VanDerPol['Xf'], mask=Config_VanDerPol['Xf_mask'] ). \
    SetLBW(lbw=Config_VanDerPol['lbw'] ). \
    SetUBW(ubw=Config_VanDerPol['ubw'] ). \
    SetLBQ(lbq=Config_VanDerPol['lbq']). \
    SetUBQ(ubq=Config_VanDerPol['ubq']). \
    SetNumberShootingNodes( Number = Config_VanDerPol['NumberShootingNodes'] ). \
    SetSolver( Solver = OptSolver.qp )
#    Build()
ocp.Build( Config_VanDerPol )

