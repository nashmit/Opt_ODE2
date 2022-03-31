from draw_cart_pendulum_experiment import PlayAnimCartAndPendulum

from casadi import *
import numpy as np
import pickle


#frame = pickle.load( open( "input.p", "rb" ) )



from DAE import ClassDefineInvertedPendulumOnACard, Config_InvertedPendulumOnACard_NMPC
from homework2 import OCP, OptSolver

InvertedPendulumOnACard_ODE = ClassDefineInvertedPendulumOnACard('InvertedPendulumOnACard_ODE')
tf = SX.sym('tf',1)
Mayer_exp = tf
#SetEndTime( tf = tf ). \
#SetEndTime( tf = Config_InvertedPendulumOnACard['tf'] ). \
#SetX_f( xf = Config_InvertedPendulumOnACard['Xf'], mask=Config_InvertedPendulumOnACard['Xf_mask'] ). \

TrueSolutionS_i_W_i_Q_i = None
max_iteration = 60
ocp = OCP('InvertedPendulumOnACard')
ActivePlotter = False
nmpc_iter = 20
Solver = OptSolver.nlp


frames = Config_InvertedPendulumOnACard_NMPC['Sn']


for i in range(0,nmpc_iter):

    #if i == nmpc_iter-1:
    #    ActivePlotter = True

    ocp.SetDAE( InvertedPendulumOnACard_ODE ). \
        SetLagrangeCostFunction( L = 0 ). \
        SetMayerCostFunction( M = Mayer_exp ). \
        SetStartTime( t0 = Config_InvertedPendulumOnACard_NMPC['t0'] ). \
        SetX_0( x0 = Config_InvertedPendulumOnACard_NMPC['Sn'],
                mask = Config_InvertedPendulumOnACard_NMPC['S0_mask'] ). \
        SetLB_Xf( eq = Config_InvertedPendulumOnACard_NMPC['Xf_lb'],
                  mask = Config_InvertedPendulumOnACard_NMPC['Xf_lb_mask'] ). \
        SetUB_Xf( eq = Config_InvertedPendulumOnACard_NMPC['Xf_ub'],
                  mask = Config_InvertedPendulumOnACard_NMPC['Xf_ub_mask'] ). \
        SetLBW( lbw = Config_InvertedPendulumOnACard_NMPC['lbw'] ). \
        SetUBW( ubw = Config_InvertedPendulumOnACard_NMPC['ubw'] ). \
        SetLBQ( lbq = Config_InvertedPendulumOnACard_NMPC['lbq'] ). \
        SetUBQ( ubq = Config_InvertedPendulumOnACard_NMPC['ubq'] ). \
        SetLB_State( eq = Config_InvertedPendulumOnACard_NMPC['lb_state'] ,
                     mask = Config_InvertedPendulumOnACard_NMPC['lb_state_mask'] ). \
        SetUB_State( eq = Config_InvertedPendulumOnACard_NMPC['ub_state'] ,
                     mask = Config_InvertedPendulumOnACard_NMPC['ub_state_mask'] ). \
        SetNumberShootingNodes( Number = Config_InvertedPendulumOnACard_NMPC['NumberShootingNodes'] ). \
        SetSolver( Solver = Solver ). \
        SetMaxIterationNumber( max_iter = max_iteration-1 ). \
        Build( Config = Config_InvertedPendulumOnACard_NMPC , TrueSolutionS_i_W_i_Q_i=TrueSolutionS_i_W_i_Q_i,
               ActivePlotter=ActivePlotter, resetInitialGuess = True )

    max_iteration = 60
    OptSolver = OptSolver.nlp
    #new_S_0 = ocp.GetEndStateOfFirstShootingInterval()
    #new_S_0 = new_S_0 + 0.01 * np.random.normal(0,1, new_S_0.size1() )
    #frames = horzcat( frames, ocp.GenerateFramesForFirstShooting25FPS() )
    frames = horzcat( frames, ocp.GetStatesAsFrames() )
    #frames = ocp.GenerateFramesForFirstShooting25FPS()
    #TrueSolutionS_i_W_i_Q_i = ocp.GetResulted_Sn_and_P()
    #TrueSolutionS_i_W_i_Q_i[0][:,0] = new_S_0
    #Config_InvertedPendulumOnACard_NMPC['Sn'] = new_S_0
    Config_InvertedPendulumOnACard_NMPC['Sn'] = \
        ocp.GetX_f_fromPreviousPhaseComputation() + \
        0.00 * np.random.normal(0,1, Config_InvertedPendulumOnACard_NMPC['Sn'].size1() )
    #Config_InvertedPendulumOnACard_NMPC['tf'] = 1
    #Config_InvertedPendulumOnACard_NMPC['tf_lb'] = 0
    #Config_InvertedPendulumOnACard_NMPC['tf_ub'] = 2
    print("Xf_lb:", ocp.GetStatesAsFrames()[:,-1] )
    #print("Sf-Xf_lb:", ocp.GetStatesAsFrames()[:,-1] - DM([0, 0, pi-pi/32, 0]) )

#print("Frames: ",frames)
#frames = ocp.GetStatesAsFrames()
#frames = horzcat( frames, frame)
PlayAnimCartAndPendulum(
    ths = frames[2,:],
    xs = frames[0,:],
    L = Config_InvertedPendulumOnACard_NMPC['q'][3],
    cart_width = Config_InvertedPendulumOnACard_NMPC['cart_width'],
    cart_height = Config_InvertedPendulumOnACard_NMPC['cart_height'],
    nrFrames = frames.size2() )

#Config_InvertedPendulumOnACard_NMPC['Sn'] = DM( ocp.GetX_f_fromPreviousPhaseComputation() )
#Config_InvertedPendulumOnACard_NMPC['tf'] = 1
#Config_InvertedPendulumOnACard_NMPC['tf_lb'] = 1
#Config_InvertedPendulumOnACard_NMPC['tf_ub'] = 5