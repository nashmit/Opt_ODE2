from casadi import *
import numpy as np
import pickle


from draw_cart_pendulum_experiment import PlayAnimCartAndPendulum

from DAE import ClassDefineInvertedPendulumOnACard, Config_InvertedPendulumOnACard
from homework2 import OCP, OptSolver


InvertedPendulumOnACard_ODE = ClassDefineInvertedPendulumOnACard('InvertedPendulumOnACard_ODE')
tf = SX.sym('tf',1)
Mayer_exp = tf
#SetEndTime( tf = tf ). \
#SetEndTime( tf = Config_InvertedPendulumOnACard['tf'] ). \
#SetX_f( xf = Config_InvertedPendulumOnACard['Xf'], mask=Config_InvertedPendulumOnACard['Xf_mask'] ). \
ocp = OCP('InvertedPendulumOnACard'). \
    SetDAE( InvertedPendulumOnACard_ODE ). \
    SetLagrangeCostFunction( L = 0 ). \
    SetMayerCostFunction( M = Mayer_exp ). \
    SetStartTime( t0 = Config_InvertedPendulumOnACard['t0'] ). \
    SetX_0( x0 = Config_InvertedPendulumOnACard['Sn'],
            mask = Config_InvertedPendulumOnACard['S0_mask'] ). \
    SetLB_Xf( eq = Config_InvertedPendulumOnACard['Xf_lb'],
              mask = Config_InvertedPendulumOnACard['Xf_lb_mask'] ). \
    SetUB_Xf( eq = Config_InvertedPendulumOnACard['Xf_ub'],
              mask = Config_InvertedPendulumOnACard['Xf_ub_mask'] ). \
    SetLBW( lbw = Config_InvertedPendulumOnACard['lbw'] ). \
    SetUBW( ubw = Config_InvertedPendulumOnACard['ubw'] ). \
    SetLBQ( lbq = Config_InvertedPendulumOnACard['lbq'] ). \
    SetUBQ( ubq = Config_InvertedPendulumOnACard['ubq'] ). \
    SetLB_State( eq = Config_InvertedPendulumOnACard['lb_state'] ,
                 mask = Config_InvertedPendulumOnACard['lb_state_mask'] ). \
    SetUB_State( eq = Config_InvertedPendulumOnACard['ub_state'] ,
                 mask = Config_InvertedPendulumOnACard['ub_state_mask'] ). \
    SetNumberShootingNodes( Number = Config_InvertedPendulumOnACard['NumberShootingNodes'] ). \
    SetSolver( Solver = OptSolver.nlp ). \
    SetMaxIterationNumber( max_iter = 80 ).\
    Build( Config=Config_InvertedPendulumOnACard , TrueSolutionS_i_W_i_Q_i=None,
           ActivePlotter=False, resetInitialGuess = True )

frames = ocp.GetStatesAsFrames()
PlayAnimCartAndPendulum(
    ths = frames[2,:],
    xs = frames[0,:],
    L = Config_InvertedPendulumOnACard['q'][3],
    cart_width = Config_InvertedPendulumOnACard['cart_width'],
    cart_height = Config_InvertedPendulumOnACard['cart_height'],
    nrFrames = frames.size2() )

Config_InvertedPendulumOnACard['Sn'] = DM( ocp.GetX_f_fromPreviousPhaseComputation() )
Config_InvertedPendulumOnACard['tf'] = 1
Config_InvertedPendulumOnACard['tf_lb'] = 1
Config_InvertedPendulumOnACard['tf_ub'] = 5
ocp.SetX_0_fromPreviousPhaseComputation(). \
    SetLB_Xf( eq = lambda state: Function( 'lambda_Xf_lb_state',
                                           [ state ], [ vertcat( ocp.GetX_f_fromPreviousPhaseComputation() ) ],
                                           ['current_state'], ['Xf_lb'] ),
              mask = [ 1 ] * len( Config_InvertedPendulumOnACard['Xf_lb_mask'] ) ). \
    SetUB_Xf( eq = lambda state: Function( 'lambda_Xf_ub_state',
                                           [ state ], [ vertcat( ocp.GetX_f_fromPreviousPhaseComputation() ) ],
                                           ['current_state'], ['Xf_ub'] ),
              mask = [ 1 ] * len( Config_InvertedPendulumOnACard['Xf_ub_mask'] ) ). \
    SetLBW( lbw = Config_InvertedPendulumOnACard['lbw'] ). \
    SetUBW( ubw = Config_InvertedPendulumOnACard['ubw'] ). \
    SetLBQ( lbq = Config_InvertedPendulumOnACard['lbq'] ). \
    SetUBQ( ubq = Config_InvertedPendulumOnACard['ubq'] ). \
    SetLB_State( eq = Config_InvertedPendulumOnACard['lb_state_phase2'] ,
                 mask = Config_InvertedPendulumOnACard['lb_state_mask_phase2'] ). \
    SetUB_State( eq = Config_InvertedPendulumOnACard['ub_state_phase2'] ,
                 mask = Config_InvertedPendulumOnACard['ub_state_mask_phase2'] ). \
    SetSolver( Solver = OptSolver.nlp ). \
    SetMaxIterationNumber( max_iter = 100 ). \
    Build( Config=Config_InvertedPendulumOnACard , TrueSolutionS_i_W_i_Q_i=None,
           ActivePlotter = False, resetInitialGuess = True )

frames = horzcat(frames, ocp.GetStatesAsFrames() )
PlayAnimCartAndPendulum(
    ths = frames[2,:],
    xs = frames[0,:],
    L = Config_InvertedPendulumOnACard['q'][3],
    cart_width = Config_InvertedPendulumOnACard['cart_width'],
    cart_height = Config_InvertedPendulumOnACard['cart_height'],
    nrFrames = frames.size2() )


#pickle.dump( horzcat( ocp.GetStatesAsFrames(), ocp.GetStatesAsFrames(), ocp.GetStatesAsFrames() ), open( "input.p", "wb" ) )
