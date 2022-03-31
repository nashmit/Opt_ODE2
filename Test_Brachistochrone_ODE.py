from PlottingSystem import OCP_Plot

import matplotlib.pyplot as plt

from DAE import *


from casadi import *
import numpy as np

from enum import Enum
from copy import deepcopy
from typing import Union


from homework2 import *

tf = SX.sym('tf',1)
Mayer_exp = tf
Brachistochrone_ODE = ClassDefineBrachistochrone("Brachistochrone")
ocp = OCP( "Optimize LotkaVolterra" ).\
    AddDAE(Brachistochrone_ODE).\
    AddLagrangeCostFunction( L = 0 ).\
    AddMayerCostFunction( M = Mayer_exp ).\
    SetStartTime( t0 = Config_Brachistochrone['t0'] ).\
    SetEndTime( tf = tf ) .\
    SetX_0( x0 = Config_Brachistochrone['Sn'], mask = Config_Brachistochrone['S0_mask'] ).\
    SetX_f( xf = Config_Brachistochrone['Xf'], mask = Config_Brachistochrone['Xf_mask'] ).\
    SetLBW( lbw = Config_Brachistochrone['lbw'] ).\
    SetUBW( ubw = Config_Brachistochrone['ubw'] ).\
    SetNumberShootingNodes( Number = Config_Brachistochrone['NumberShootingNodes'] ).\
    SetSolver( Solver = OptSolver.qp )
ocp.Build(Config_Brachistochrone)
