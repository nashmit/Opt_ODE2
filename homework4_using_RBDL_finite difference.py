from casadi import *
import numpy as np
import pickle


from draw_cart_pendulum_experiment import PlayAnimCartAndPendulum

#from DAE import ClassDefineInvertedPendulumOnACard, Config_InvertedPendulumOnACard
from homework4_DAE_def import ClassDefinePendulumCart_Using_RBDL_as_ODE, Config_InvertedPendulumOnACard_usingRBDL_as_ODE
from homework2 import OCP, OptSolver


#InvertedPendulumOnACard_ODE = ClassDefineInvertedPendulumOnACard('InvertedPendulumOnACard_ODE')
InvertedPendulumOnACard_RBDL_as_ODE = ClassDefinePendulumCart_Using_RBDL_as_ODE('InvertedPendulumOnACard_RBDL_as_ODE')

tf = MX.sym('tf',1)
Mayer_exp = tf
#SetEndTime( tf = tf ). \
#SetEndTime( tf = Config_InvertedPendulumOnACard['tf'] ). \
#SetX_f( xf = Config_InvertedPendulumOnACard['Xf'], mask=Config_InvertedPendulumOnACard['Xf_mask'] ). \
ocp = OCP('InvertedPendulumOnACard'). \
    SetDAE( InvertedPendulumOnACard_RBDL_as_ODE ). \
    SetLagrangeCostFunction( L = 0 ). \
    SetMayerCostFunction( M = Mayer_exp ). \
    SetStartTime( t0 = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['t0'] ). \
    SetX_0( x0 = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['Sn'],
            mask = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['S0_mask'] ). \
    SetLB_Xf( eq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['Xf_lb'],
              mask = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['Xf_lb_mask'] ). \
    SetUB_Xf( eq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['Xf_ub'],
              mask = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['Xf_ub_mask'] ). \
    SetLBW( lbw = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['lbw'] ). \
    SetUBW( ubw = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['ubw'] ). \
    SetLBQ( lbq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['lbq'] ). \
    SetUBQ( ubq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['ubq'] ). \
    SetLB_State( eq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['lb_state'] ,
                 mask = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['lb_state_mask'] ). \
    SetUB_State( eq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['ub_state'] ,
                 mask = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['ub_state_mask'] ). \
    SetNumberShootingNodes( Number = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['NumberShootingNodes'] ). \
    SetSolver( Solver = OptSolver.nlp ). \
    SetMaxIterationNumber( max_iter = 0 ). \
    Build( Config=Config_InvertedPendulumOnACard_usingRBDL_as_ODE , TrueSolutionS_i_W_i_Q_i=None,
           ActivePlotter=True, resetInitialGuess = True )

exit(0) # this exit exists for simulation only. ( also, max_iter=0 ) and tf=2, also 'Sn': DM([1, 0, 1, 0]) also ActivePlotter=True

frames = ocp.GetStatesAsFrames()
PlayAnimCartAndPendulum(
    ths = frames[2,:],
    xs = frames[0,:],
    L = 1.5,
    cart_width = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['cart_width'],
    cart_height = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['cart_height'],
    nrFrames = frames.size2() )

Config_InvertedPendulumOnACard_usingRBDL_as_ODE['Sn'] = DM( ocp.GetX_f_fromPreviousPhaseComputation() )
Config_InvertedPendulumOnACard_usingRBDL_as_ODE['tf'] = 1
Config_InvertedPendulumOnACard_usingRBDL_as_ODE['tf_lb'] = 1
Config_InvertedPendulumOnACard_usingRBDL_as_ODE['tf_ub'] = 5
ocp.SetX_0_fromPreviousPhaseComputation(). \
    SetLB_Xf( eq = lambda state: Function( 'lambda_Xf_lb_state',
                                           [ state ], [ vertcat( ocp.GetX_f_fromPreviousPhaseComputation() ) ],
                                           ['current_state'], ['Xf_lb'] ),
              mask = [ 1 ] * len( Config_InvertedPendulumOnACard_usingRBDL_as_ODE['Xf_lb_mask'] ) ). \
    SetUB_Xf( eq = lambda state: Function( 'lambda_Xf_ub_state',
                                           [ state ], [ vertcat( ocp.GetX_f_fromPreviousPhaseComputation() ) ],
                                           ['current_state'], ['Xf_ub'] ),
              mask = [ 1 ] * len( Config_InvertedPendulumOnACard_usingRBDL_as_ODE['Xf_ub_mask'] ) ). \
    SetLBW( lbw = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['lbw'] ). \
    SetUBW( ubw = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['ubw'] ). \
    SetLBQ( lbq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['lbq'] ). \
    SetUBQ( ubq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['ubq'] ). \
    SetLB_State( eq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['lb_state_phase2'] ,
                 mask = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['lb_state_mask_phase2'] ). \
    SetUB_State( eq = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['ub_state_phase2'] ,
                 mask = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['ub_state_mask_phase2'] ). \
    SetSolver( Solver = OptSolver.nlp ). \
    SetMaxIterationNumber( max_iter = 80 ). \
    Build( Config=Config_InvertedPendulumOnACard_usingRBDL_as_ODE , TrueSolutionS_i_W_i_Q_i=None,
           ActivePlotter = False, resetInitialGuess = True )

frames = horzcat(frames, ocp.GetStatesAsFrames() )
PlayAnimCartAndPendulum(
    ths = frames[2,:],
    xs = frames[0,:],
    L = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['q'][3],
    cart_width = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['cart_width'],
    cart_height = Config_InvertedPendulumOnACard_usingRBDL_as_ODE['cart_height'],
    nrFrames = frames.size2() )


#pickle.dump( horzcat( ocp.GetStatesAsFrames(), ocp.GetStatesAsFrames(), ocp.GetStatesAsFrames() ), open( "input.p", "wb" ) )
