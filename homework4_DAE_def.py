import numpy as np
from casadi import *
from homework4_def_cart_inverted_pendulum import Box2d_world_Init_cart_inverted_pendulum

import biorbd
from FeatherstoneDynamics import *


# Nice implementation / integration for pytorch:
# https://groups.google.com/g/casadi-users/c/gLJNzajFM6w/m/wBCNzzm7AQAJ

# https://github.com/casadi/casadi/blob/develop/docs/examples/python/callback.py

class Simulation_Interface(Callback):

    def __init__(self, name, world, SparsityX, SparsityP, opts={} ):

        Callback.__init__(self)

        self.world = world
        self.SparsityX = SparsityX
        self.SparsityP = SparsityP

        # extract other options...!
        self.dt = opts['dt']

        # initialize self.world based on current options!

        self.construct(name, { "enable_fd" : True } )

    # Number of inputs and outputs
    def get_n_in(self):
        return 2 # number of parameters
    def get_n_out(self):
        return 1 # size of state space
        # return 2 # { xf and qf }

    def get_name_in(self, indx):
        switcher = {
            0:'x0',
            1:'p',
        }
        return switcher.get(indx, 'wrong idx nr' )
    def get_name_out(self, indx):
        switcher = {
            0:'xf', # newX
            # 1:'qf'
        }
        return switcher.get(indx, 'wrong idx nr' )

    def get_sparsity_in(self, indx):
        switcher = {
            0: self.SparsityX, #Sparsity.dense(1),
            1: self.SparsityP, #Sparsity.dense(2),
        }
        return switcher.get(indx, 'wrong idx nr' )
    def get_sparsity_out(self, indx):
        switcher = {
            0: self.SparsityX, # same sparsity as input 'x0'
            # 1: Sparsity_qf
        }
        return switcher.get(indx, 'wrong idx nr' )

    # Initialize the object
    def init(self):
        print('initializing object(de fault)')

    # Evaluate numerically
    def eval(self, arg):

        assert False, "eval() was not implemented! ( please, implement this interface! ) "
        # this is the method one must implement
        # to initialize the simulation by providing "x0" and "p" as "arg" input

        pass

class ODE_Interface(Callback):

    def __init__(self, name, SparsityX, SparsityP,
                 opts={
                     "enable_fd" : True,
                     'enable_forward' : False, 'enable_reverse' : False, 'enable_jacobian' : False,
                     #'verbose': True,
                     #'print_time' : True
                 } ):

        Callback.__init__(self)

        self.SparsityX = SparsityX.sparsity()
        self.SparsityP = SparsityP.sparsity()

        # extract other options...!

        self.construct(name, opts )

    # Number of inputs and outputs
    def get_n_in(self):
        return 2 # number of parameters
    def get_n_out(self):
        return 1 # the state space variable ( as one, multidimensional variable )

    def get_name_in(self, indx):
        switcher = {
            0:'x0',
            1:'p',
        }
        return switcher.get(indx, 'wrong idx nr' )
    def get_name_out(self, indx):
        switcher = {
            0:'xdot', # where 'x' is the 'state' variable
        }
        return switcher.get(indx, 'wrong idx nr' )

    def get_sparsity_in(self, indx):
        switcher = {
            0: self.SparsityX, #Sparsity.dense(1),
            1: self.SparsityP, #Sparsity.dense(2),
        }
        return switcher.get(indx, 'wrong idx nr' )
    def get_sparsity_out(self, indx):
        switcher = {
            0: self.SparsityX, # same sparsity as input 'x' state variable
        }
        return switcher.get(indx, 'wrong idx nr' )

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):

        assert False, "eval() was not implemented! ( please, implement this interface! ) "
        # this is the method one must implement
        # to initialize the simulation by providing "x0" and "p" as "arg" input

        pass

# Sim = Simulation( name='Sim', world=None, SparsityX=Sparsity.dense(1), SparsityP=Sparsity.dense(2) )
# Sim(5,[1,2])
#
#
# x = MX.sym('x',1)
# p = MX.sym('p',2)
#
# exp_xdot = Sim(x,p)
# print( exp_xdot )
#
# Sim.call([1,[1,2]])
#
# f_test = Function('f_test', [ x, p ], [ exp_xdot ], ['x','p'],['out'], ) #{'print_in':True} )
# print(f_test)
#
# f_test.call([2,[1,2]])
# f_test(2,[1,2])
# print(jacobian(f_test(x,p),x))
# #f_test.print_options()

def simple_World_Init():

    from Box2D import (b2PolygonShape, b2World)

    world = b2World()  # default gravity is (0,-10) and doSleep is True
    groundBody = world.CreateStaticBody(position=(0, -10),
                                        shapes=b2PolygonShape(box=(50, 10)),
                                        )

    # Create a dynamic body at (0, 4)
    body = world.CreateDynamicBody(position=(0, 4))

    # And add a box fixture onto it (with a nonzero density, so it will move)
    box = body.CreatePolygonFixture(box=(1, 1), density=1, friction=0.3)

    return world
    pass


from DAE import DAEInterface

class ClassDefinePendulum_Using_BOX2D_as_ODE(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):

        class Simulation(Simulation_Interface):

            def eval(self, arg):

                #this is the place where the initialization of the simulation happens
                #by providing "x0" and "p" as "arg" input

                #print('inside eval:')
                #print('eval arg:', arg)

                ####
                # init
                world = self.world
                # world.step(dt, ... )
                # return new state spate [X]
                ####

                # output must be return as list e.g.: "[ ]", same as in a regular Casadi function!
                return  [ arg[0] ] #[World]    #[ arg[0] ]  #[ self.world ]

                pass

        x = MX.sym('x',1) # state space

        w = MX.sym('w',1) # control parameter
        q = MX.sym('q',1) # fixed constant parameters for all shooting nodes
        p = vertcat( w, q) # complete vector of parameters

        DynamicalSystem = Simulation

        #world = None # for the moment... I must populate this variable with a real Box2d environment.
        #world = simple_World_Init()
        world = Box2d_world_Init_cart_inverted_pendulum()

        dae = { 'x': x, 'p': p, 'DynamicalSystem': DynamicalSystem, 'world': world }

        self.x = x
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.p = p
        self.xdot = None # we don't have xdot but a numerical simulation
        self.dae = dae # no symbolical ode/dae this time
        self.DynamicalSystem = DynamicalSystem # rapper for Box2D simulation



class ClassDefinePendulumCart_Using_RBDL_as_ODE(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)

        self.options = self.options | {
            'enable_fd' : True,
            'enable_forward' : False, 'enable_reverse' : False,
            'enable_jacobian' : False,
            'fd_method' : 'smoothing'
        }
        # 'dump_in' : True, 'dump_out' : True

        pass

    def GetIntgType(self):
        return 'cvodes' # rk / collocation / idas / cvodes
        pass

    def DefineDAE(self):

        class Simulation(ODE_Interface):

            # Initialize the object
            def init(self):
                print('initializing ("pendulumCart" model loading)')
                if not hasattr(self,'model'):
                    self.model = biorbd.Model('finalProj_Input/pendulumCart.bioMod')

            def eval(self, arg):

                state = arg[0].full().T[0] #reshape state variable from DM to py-list

                Q = np.array([ state[0], state[2] ])
                QDot = np.array([ state[1], state[3] ])

                states = self.model.stateSet()
                for state in states:
                    state.setActivation( 0 )
                joint_torque = self.model.muscularJointTorque( states, Q, QDot )

                control = arg[1].full()[0][0]
                external_force_vector = \
                    np.array([ 0, 0, 0, control, control, control ]) # Spatial vector has 6 components
                f_ext = biorbd.VecBiorbdSpatialVector()
                f_ext.append(biorbd.SpatialVector(external_force_vector))

                # Compute the acceleration of the model due to these torques
                QDDot = self.model.ForwardDynamics( Q, QDot, np.array([control,0]) ) # I shuld consider the control parameter as
                # external force but it seems that the external force has no effect over the model

                #QDDot = DM(QDDot.to_array()) + vertcat( arg[1], 0 ) # add control
                QDDot = QDDot.to_array()
                #print("QDDot: ",QDDot.to_array())

                # ( rDot, rDDot, thetaDot, thetaDDot )
                return [ vertcat( QDot[0], QDDot[0], QDot[1], QDDot[1] ) ]

                #this is the place where the initialization of the simulation happens
                #by providing "x0" and "p" as "arg" input

                #print('inside eval:')
                #print('eval arg:', arg)

                ####
                # init
                #world = self.world
                # world.step(dt, ... )
                # return new state spate [X]
                ####

                # output must be return as list e.g.: "[ ]", same as in a regular Casadi function!
                #return  [ arg[0] ] #[World]    #[ arg[0] ]  #[ self.world ]

                pass

        #r = SX.sym( 'r', 1 )
        #rDot = SX.sym( 'rDot', 1 )
        #theta = SX.sym( 'theta', 1 )
        #thetaDot = SX.sym( 'thetaDot', 1 )
        x = MX.sym('x',4) # state space

        w = MX.sym('w',1) # control parameter
        q = MX.sym('q',0) # fixed constant parameters for all shooting nodes
        p = vertcat( w, q) # complete vector of parameters

        self.Sim = Simulation('xdot', x, p)
        xdot = self.Sim(x0=x,p=p)['xdot']

        ode = {'x': x, 'p': p, 'ode': xdot}

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.xdot = xdot
        self.dae = ode


class ClassDefinePendulum_Using_RBDL_as_ODE(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)

        self.options = self.options | {
            'enable_fd' : True,
            'enable_forward' : False, 'enable_reverse' : False,
            'enable_jacobian' : False,
            'fd_method' : 'smoothing'
        }
        # 'dump_in' : True, 'dump_out' : True

        pass

    def GetIntgType(self):
        return 'cvodes' # rk / collocation / idas / cvodes
        pass

    def DefineDAE(self):

        class Simulation(ODE_Interface):

            # Initialize the object
            def init(self):
                print('initializing ("pendulum" model loading)')
                if not hasattr(self,'model'):
                    self.model = biorbd.Model('finalProj_Input/pendulum.bioMod')

            def eval(self, arg):

                state = arg[0].full().T[0] #reshape state variable from DM to py-list

                Q = np.array([ state[0] ])
                QDot = np.array([ state[1] ])

                states = self.model.stateSet()
                for state in states:
                    state.setActivation( 0 )
                joint_torque = self.model.muscularJointTorque( states, Q, QDot )

                control = arg[1].full()[0][0]
                external_force_vector = \
                    np.array([ 0, 0, 0, control, control, control ]) # Spatial vector has 6 components
                f_ext = biorbd.VecBiorbdSpatialVector()
                f_ext.append(biorbd.SpatialVector(external_force_vector))

                # Compute the acceleration of the model due to these torques
                QDDot = self.model.ForwardDynamics( Q, QDot, np.array([control]) ) # I shuld consider the control parameter as
                # external force but it seems that the external force has no effect over the model

                #QDDot = DM(QDDot.to_array()) + vertcat( arg[1], 0 ) # add control
                QDDot = QDDot.to_array()
                #print("QDDot: ",QDDot.to_array())

                # ( rDot, rDDot, thetaDot, thetaDDot )
                return [ vertcat( QDot[0], QDDot[0] ) ]

                #this is the place where the initialization of the simulation happens
                #by providing "x0" and "p" as "arg" input

                #print('inside eval:')
                #print('eval arg:', arg)

                ####
                # init
                #world = self.world
                # world.step(dt, ... )
                # return new state spate [X]
                ####

                # output must be return as list e.g.: "[ ]", same as in a regular Casadi function!
                #return  [ arg[0] ] #[World]    #[ arg[0] ]  #[ self.world ]

                pass

        #####r = SX.sym( 'r', 1 )
        #####rDot = SX.sym( 'rDot', 1 )
        #theta = SX.sym( 'theta', 1 )
        #thetaDot = SX.sym( 'thetaDot', 1 )
        x = MX.sym('x',2) # state space

        w = MX.sym('w',1) # control parameter
        q = MX.sym('q',0) # fixed constant parameters for all shooting nodes
        p = vertcat( w, q) # complete vector of parameters

        self.Sim = Simulation('xdot', x, p)
        xdot = self.Sim(x0=x,p=p)['xdot']

        ode = {'x': x, 'p': p, 'ode': xdot}

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.xdot = xdot
        self.dae = ode


class ClassDefinePendulumCart_Using_SymFederstone_as_ODE(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)

        self.options = self.options | {
            #'enable_fd' : True,
            #'enable_forward' : False, 'enable_reverse' : False,
            #'enable_jacobian' : False,
            #'fd_method' : 'smoothing'
        }
        # 'dump_in' : True, 'dump_out' : True

        pass

    def GetIntgType(self):
        return 'cvodes' # rk / collocation / idas / cvodes
        pass

    def LoadFrom_URDF(self):

        urdf_path = "finalProj_Input/pendulumCart.urdf"
        rootLink = "link Root"
        endLink = "link B"

        robot = Robot()
        robot.Load(urdf_path)
        #n_joints = robot.getJointsNumber(rootLink, endLink)
        gravity = [0, 0, -9.81]
        qddot_sym_exp, q, q_dot, tau = robot.GetForwardDynamics_CRBA_exp(rootLink, endLink, gravity = gravity)

        # make control for the second DOF inaccessible ( build underactuated system )
        qddot_sym_exp = substitute(qddot_sym_exp, tau[1] ,0)

        # ode( system of first order ode, size=4X1 ), x( size=4X1 ), w( size=1 ), q( size=0 )
        return \
            vertcat(
                q_dot[0],
                qddot_sym_exp[0],
                q_dot[1],
                qddot_sym_exp[1] ),\
            vertcat(
                q[0],
                q_dot[0],
                q[1],
                q_dot[1] ), \
            tau[0], \
            SX.sym('q', 0)
        pass

    def DefineDAE(self):

        class Simulation(ODE_Interface):

            # Initialize the object
            def init(self):
                print('initializing ("pendulumCart" model loading)')
                if not hasattr(self,'model'):
                    self.model = biorbd.Model('finalProj_Input/pendulumCart.bioMod')

            def eval(self, arg):

                state = arg[0].full().T[0] #reshape state variable from DM to py-list

                Q = np.array([ state[0], state[2] ])
                QDot = np.array([ state[1], state[3] ])

                states = self.model.stateSet()
                for state in states:
                    state.setActivation( 0 )
                joint_torque = self.model.muscularJointTorque( states, Q, QDot )

                control = arg[1].full()[0][0]
                external_force_vector = \
                    np.array([ 0, 0, 0, control, control, control ]) # Spatial vector has 6 components
                f_ext = biorbd.VecBiorbdSpatialVector()
                f_ext.append(biorbd.SpatialVector(external_force_vector))

                # Compute the acceleration of the model due to these torques
                QDDot = self.model.ForwardDynamics( Q, QDot, np.array([control,0]) ) # I shuld consider the control parameter as
                # external force but it seems that the external force has no effect over the model

                #QDDot = DM(QDDot.to_array()) + vertcat( arg[1], 0 ) # add control
                QDDot = QDDot.to_array()
                #print("QDDot: ",QDDot.to_array())

                # ( rDot, rDDot, thetaDot, thetaDDot )
                return [ vertcat( QDot[0], QDDot[0], QDot[1], QDDot[1] ) ]

                #this is the place where the initialization of the simulation happens
                #by providing "x0" and "p" as "arg" input

                #print('inside eval:')
                #print('eval arg:', arg)

                ####
                # init
                #world = self.world
                # world.step(dt, ... )
                # return new state spate [X]
                ####

                # output must be return as list e.g.: "[ ]", same as in a regular Casadi function!
                #return  [ arg[0] ] #[World]    #[ arg[0] ]  #[ self.world ]

                pass

        #r = SX.sym( 'r', 1 )
        #rDot = SX.sym( 'rDot', 1 )
        #theta = SX.sym( 'theta', 1 )
        #thetaDot = SX.sym( 'thetaDot', 1 )
        #x = MX.sym('x',4) # state space

        #w = MX.sym('w',1) # control parameter
        #q = MX.sym('q',0) # fixed constant parameters for all shooting nodes

        xdot, x, w, q = self.LoadFrom_URDF()

        p = vertcat( w, q) # complete vector of parameters

        #self.Sim = Simulation('xdot', x, p)
        #xdot = self.Sim(x0=x,p=p)['xdot']

        ode = {'x': x, 'p': p, 'ode': xdot}

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.xdot = xdot
        self.dae = ode


def integratorOnTopOf_Box2D( name, Intg_name, Simulation, opts):

    # must take into consideration that the 'opts' set might contain a list of possible time locations !!!
    # default [0,1]
    t0 = opts['t0']
    tf = opts['tf']

    frequency = 100  # Hz
    time = tf - t0
    nr_steps = time / ( 1 / frequency )
    assert nr_steps >= 1.0, "Simulation 'Time interval' smaller than 1/frequency!"

    # Simulation.quad

    #  name, world, SparsityX, SparsityP, opts={}
    SingleStep = Simulation['DynamicalSystem'](
        name='SingleStep',
        world = Simulation['world'],
        SparsityX = Simulation['x'].sparsity(),
        SparsityP = Simulation['p'].sparsity(),
        opts = {'dt': 1/frequency}
    )

    intg = SingleStep.fold( int(nr_steps) )

    I = Function(
        'I',
        [ Simulation['x'], Simulation['p'] ],
        [ intg( Simulation['x'], Simulation['p'] ) ],
        [ 'x0', 'p' ],
        [ 'xf' ]
    )

    return I

    pass

def TestUsingBox2D():
    Box2dSim = ClassDefinePendulum_Using_BOX2D_as_ODE("Box2DSim")

    intg_Box2D = integratorOnTopOf_Box2D(
        name='I',
        Intg_name='myIntegrator',
        Simulation=Box2dSim.dae,
        opts={'t0':0, 'tf':1}
    )
    print(intg_Box2D)

Config_InvertedPendulumOnACard_usingRBDL_as_ODE={
    't0': 0,
    'tf': 2,
    'tf_lb': 1,
    'tf_ub':100,
    'NumberShootingNodes': 20,#20,#50,
    'Sn': DM([0, 1, 1, 0]), # position and velocity at t_0
    'S0_mask': [1, 1, 1, 1],
    #'S0_lb': [0, 0, 0, 0],
    #'S0_ub': [0, 0, 0, 0],
    'SnName': [ 'r', 'rDot', 'theta', 'thetaDot' ],
    #'Xf': [0, 0, pi, 0], # 'position' and 'velocity' at tf of each DOF
    #'Xf_mask': [0, 0, 1, 1],#[0, 0],#[1, 1], ->final velocity is not that important
    'Xf_lb': lambda state: Function(
        'lambda_Xf_lb_state',
        [ state ], [ vertcat( 0, 0, pi-pi/32, 0 ) ],
        ['current_state'], ['Xf_lb'] ), # lower bound for 'theta' and 'thetaDot'
    'Xf_lb_mask': [0, 0, 1, 1],
    'Xf_ub': lambda state: Function(
        'lambda_Xf_ub_state',
        [ state ], [ vertcat( 0, 0, pi-pi/32, 0 ) ],
        ['current_state'], ['Xf_ub'] ), # upper bound for 'theta' and 'thetaDot'
    'Xf_ub_mask': [0, 0, 1, 1],
    'w': [ 0 ], # doesn't influence the ODE
    'wName': ['u_r'], #control
    'lbw': [ -20 ],#[-20],
    'ubw': [  20 ],#[20],
    'q': [ ],
    'qName': [ ],
    'lbq': [ ],
    'ubq': [ ],
    'cart_width': 0.5,
    'cart_height': 0.3,
    'lb_state' :
        lambda state: Function(
            'lambda_lb_state',
            [ state ], [ vertcat( -0.7, -2.2, 0, 0 ) ],
            ['current_state'], ['lb_state'] ), # lower bound for 'r' and velocity norm
    'lb_state_mask': [ 1, 1, 0, 0 ], # lower bound mask for 'r' and velocity norm
    'ub_state' :
        lambda state: Function(
            'lambda_ub_state',
            [ state ], [ vertcat(  0.7,  2.2, 0, 0 ) ],
            ['current_state'], ['ub_state'] ), # upper bound for 'r' and velocity norm
    'ub_state_mask': [ 1, 1, 0, 0 ], # upper bound mask for 'r' and velocity norm

    'lb_state_phase2' :
        lambda state: Function(
            'lambda_lb_state',
            [ state ], [ vertcat( -0.7, -2.2, pi-pi/32, 0 ) ],
            ['current_state'], ['lb_state'] ), # lower bound for 'r' and velocity norm
    'lb_state_mask_phase2': [ 1, 1, 1, 0 ], # lower bound mask for 'r' and velocity norm
    'ub_state_phase2' :
        lambda state: Function(
            'lambda_ub_state',
            [ state ], [ vertcat(  0.7,  2.2, pi+pi/32, 0 ) ],
            ['current_state'], ['ub_state'] ), # upper bound for 'r' and velocity norm
    'ub_state_mask_phase2': [ 1, 1, 1, 0 ], # upper bound mask for 'r' and velocity norm
}

Config_InvertedPendulumOnACard_Using_SymFederstone_as_ODE={
    't0': 0,
    'tf': 10,
    'tf_lb': 1,
    'tf_ub':100,
    'NumberShootingNodes': 20,#20,#50,
    'Sn': DM([0, 0, 0, 0]), # position and velocity at t_0
    'S0_mask': [1, 1, 1, 1],
    #'S0_lb': [0, 0, 0, 0],
    #'S0_ub': [0, 0, 0, 0],
    'SnName': [ 'r', 'rDot', 'theta', 'thetaDot' ],
    #'Xf': [0, 0, pi, 0], # 'position' and 'velocity' at tf of each DOF
    #'Xf_mask': [0, 0, 1, 1],#[0, 0],#[1, 1], ->final velocity is not that important
    'Xf_lb': lambda state: Function(
        'lambda_Xf_lb_state',
        [ state ], [ vertcat( 0, 0, pi-pi/32, 0 ) ],
        ['current_state'], ['Xf_lb'] ), # lower bound for 'theta' and 'thetaDot'
    'Xf_lb_mask': [0, 0, 1, 1],
    'Xf_ub': lambda state: Function(
        'lambda_Xf_ub_state',
        [ state ], [ vertcat( 0, 0, pi-pi/32, 0 ) ],
        ['current_state'], ['Xf_ub'] ), # upper bound for 'theta' and 'thetaDot'
    'Xf_ub_mask': [0, 0, 1, 1],
    'w': [ 0 ], # doesn't influence the ODE
    'wName': ['u_r'], #control
    'lbw': [ -20 ],#[-20],
    'ubw': [  20 ],#[20],
    'q': [ ],
    'qName': [ ],
    'lbq': [ ],
    'ubq': [ ],
    'cart_width': 0.5,
    'cart_height': 0.3,
    'lb_state' :
        lambda state: Function(
            'lambda_lb_state',
            [ state ], [ vertcat( -0.7, -2.2, 0, 0 ) ],
            ['current_state'], ['lb_state'] ), # lower bound for 'r' and velocity norm
    'lb_state_mask': [ 1, 1, 0, 0 ], # lower bound mask for 'r' and velocity norm
    'ub_state' :
        lambda state: Function(
            'lambda_ub_state',
            [ state ], [ vertcat(  0.7,  2.2, 0, 0 ) ],
            ['current_state'], ['ub_state'] ), # upper bound for 'r' and velocity norm
    'ub_state_mask': [ 1, 1, 0, 0 ], # upper bound mask for 'r' and velocity norm

    'lb_state_phase2' :
        lambda state: Function(
            'lambda_lb_state',
            [ state ], [ vertcat( -0.7, -2.2, pi-pi/32, 0 ) ],
            ['current_state'], ['lb_state'] ), # lower bound for 'r' and velocity norm
    'lb_state_mask_phase2': [ 1, 1, 1, 0 ], # lower bound mask for 'r' and velocity norm
    'ub_state_phase2' :
        lambda state: Function(
            'lambda_ub_state',
            [ state ], [ vertcat(  0.7,  2.2, pi+pi/32, 0 ) ],
            ['current_state'], ['ub_state'] ), # upper bound for 'r' and velocity norm
    'ub_state_mask_phase2': [ 1, 1, 1, 0 ], # upper bound mask for 'r' and velocity norm
}


if __name__ == '__main__':

    ################## Inverted Pendulum using RBDL and finite difference: tests ##################

    # InvertedPendulumOnACard_RBDL_as_ODE = \
    #     ClassDefinePendulumCart_Using_RBDL_as_ODE('InvertedPendulumOnACard_RBDL_as_ODE')

    # print("ODE: ", InvertedPendulumOnACard_RBDL_as_ODE )
    # print("xdot: ", InvertedPendulumOnACard_RBDL_as_ODE.xdot )
    # xdot = Function('xdot',
    #                symvar(InvertedPendulumOnACard_RBDL_as_ODE.xdot),
    #                [InvertedPendulumOnACard_RBDL_as_ODE.xdot],
    #                ['x0','p'],
    #                ['ode']
    #                )
    # print(xdot)
    # print("xdot(x0=[0,0,0,0],p=0): ", xdot( x0=[0,0,0,0], p=0 )['ode'] )
    # print("xdot(x0=[0,0,0,0],p=0.1): ", xdot( x0=[0,0,0,0], p=0.1 )['ode'] )
    # print("X: ",InvertedPendulumOnACard_RBDL_as_ODE.GetX())
    # print("P(w,q): ",InvertedPendulumOnACard_RBDL_as_ODE.GetP())
    # jac_xdot = jacobian(InvertedPendulumOnACard_RBDL_as_ODE.xdot,
    #                     vertcat(
    #                         InvertedPendulumOnACard_RBDL_as_ODE.GetX(),
    #                         InvertedPendulumOnACard_RBDL_as_ODE.GetP()
    #                     ))
    # print("jac: ", jac_xdot)
    # print("jac_xdot shape: ", jac_xdot.shape)

    # I = integrator(
    #     'I',
    #     'cvodes',
    #     InvertedPendulumOnACard_RBDL_as_ODE.dae,
    #     {
    #         't0': 0, 'tf': 1,
    #         'enable_fd' : True,
    #         'enable_forward' : False, 'enable_reverse' : False, 'enable_jacobian' : False,
    #         #'fd_method' : 'smoothing',
    #         #'print_in':True,                   #print input value parameters(debug)
    #         #'print_out':True,                  #print output value parameters(debug)
    #         #'show_eval_warnings':True,         #(debug)
    #         #'inputs_check':True,               #(debug)
    #         #'regularity_check':True,           #(debug)
    #         #'verbose':True,                    #(debug)
    #         #'print_time' : True,               #(debug)
    #         #'simplify':True, 'expand':True,
    #         #'number_of_finite_elements': 5     # rk
    #         #'max_order' : 4                    #idas/cvodes 2/3/4/5
    #         #'max_multistep_order': 5           #idas/cvodes 2..5
    #         #'dump_in' : True,                  #(debug)
    #         #'dump_out' : True,                 #(debug)
    #     }
    # )
    # print("I (intg):", I )
    #
    # var1 = I( x0=[0, 0, 0, 0], p=0 )['xf']
    # var2 = I( x0=[0, 0, 0, 0], p=0.0215443 )['xf']
    # print("I ( evaluation at 'x0=[0,0,0,0], p=0' ):", var1 )
    # print("I ( evaluation at 'x0=[0,0,0,0], p=0.0215443' ):", var2 )
    # print("var1 - va2: ", var1 - var2)
    #
    # I_accum = I.mapaccum(
    #     5,
    #     {
    #         #'verbose': True,
    #         'print_time' : True
    #     }
    # )
    # acum1 = I_accum( x0=[0,0,0,0], p=0 )['xf']
    # acum2 = I_accum( x0=[0,0,0,0], p=0.0215443 )['xf']
    # print('I_accum:', I_accum )
    # print("I_accum evaluation at '( x0=[0,0,0,0], p=0 )':", acum1 )
    # print("I_accum evaluation at '( x0=[0,0,0,0], p=0.0215443 )':", acum2 )
    # print("acum1-acum2: ",acum1 - acum2 )
    #
    #
    # x0Var = MX.sym('x0Var', InvertedPendulumOnACard_RBDL_as_ODE.GetX().shape)
    # pVar = MX.sym('pVar', InvertedPendulumOnACard_RBDL_as_ODE.GetP().shape)
    #
    # I_exp = I( x0=x0Var, p=pVar )
    # print("I_exp:", I_exp )
    #
    # I_exp_jac = jacobian( I_exp['xf'],
    #                       vertcat(
    #                           x0Var,
    #                           pVar
    #                       )
    #                     )
    # print("I_exp_jac:", I_exp_jac )
    #
    # F_I_exp_jac = Function(
    #     'F_I_exp_jac',
    #     [
    #         x0Var,
    #         pVar
    #     ],
    #     [ I_exp_jac ],
    #     ['x0','p'],
    #     ['dxf_dx0dp'],
    #     {
    #         #'verbose': True,
    #         'print_time' : True
    #     }
    # )
    # print("F_I_exp_jac:", F_I_exp_jac )
    #
    # print("F_I_exp_jac evaluation at '( [0, 0, 0, 0], 0 )':", F_I_exp_jac( [0, 0, 0, 0], 0 ) )
    #
    # hess = jacobian( jacobian( I_exp['xf'], vertcat(x0Var, pVar)), vertcat(x0Var, pVar) )
    # F_I_exp_hessian = Function(
    #     'F_I_exp_hessian',
    #     [ x0Var, pVar],
    #     [ hess ],
    #     ['x0', 'p'],
    #     ['dxf^{2}_dx0dp_{i}_dx0dp_{j}'],
    #     {
    #         #'verbose': True,
    #         'print_time' : True
    #     }
    # )
    # print("Hessian at '( [0, 0, 0, 0], 0 )':", F_I_exp_hessian(x0=[0, 0, 0, 0], p=0))


########################## -> pendulum ( simulation ) using RBDL ###################
    #
    #
    # Pendulum_using_RBDL_as_ODE = \
    #     ClassDefinePendulum_Using_RBDL_as_ODE("Pendulum_using_RBDL_as_ODE")
    #
    # print("ODE: ", Pendulum_using_RBDL_as_ODE )
    # print("xdot: ", Pendulum_using_RBDL_as_ODE.xdot )
    # xdot = Function('xdot',
    #                 symvar(Pendulum_using_RBDL_as_ODE.xdot),
    #                 [Pendulum_using_RBDL_as_ODE.xdot],
    #                 ['x0','p'],
    #                 ['ode']
    #                 )
    # print(xdot)
    # print("xdot(x0=[0,0],p=0): ", xdot( x0=[0,0], p=0 )['ode'] )
    # print("xdot(x0=[0,0],p=0.1): ", xdot( x0=[0,0], p=0.1 )['ode'] )
    # print("X: ",Pendulum_using_RBDL_as_ODE.GetX())
    # print("P(w,q): ",Pendulum_using_RBDL_as_ODE.GetP())
    # jac_xdot = jacobian(Pendulum_using_RBDL_as_ODE.xdot,
    #                     vertcat(
    #                         Pendulum_using_RBDL_as_ODE.GetX(),
    #                         Pendulum_using_RBDL_as_ODE.GetP()
    #                     ))
    # print("jac: ", jac_xdot)
    # print("jac_xdot shape: ", jac_xdot.shape)
    #
    # I = integrator(
    #     'I',
    #     'cvodes', #cvodes/idas/rk/collocation
    #     Pendulum_using_RBDL_as_ODE.dae,
    #     {
    #         #'t0': 0, 'tf': 1,
    #         'grid': [ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ],
    #         'enable_fd' : True,
    #         'enable_forward' : False, 'enable_reverse' : False, 'enable_jacobian' : False,
    #         #'fd_method' : 'smoothing',
    #         #'print_in':True,                   #print input value parameters(debug)
    #         #'print_out':True,                  #print output value parameters(debug)
    #         #'show_eval_warnings':True,         #(debug)
    #         #'inputs_check':True,               #(debug)
    #         #'regularity_check':True,           #(debug)
    #         #'verbose':True,                    #(debug)
    #         #'print_time' : True,               #(debug)
    #         #'simplify':True, 'expand':True,
    #         #'number_of_finite_elements': 5     # rk
    #         #'max_order' : 4                    #idas/cvodes 2/3/4/5
    #         #'max_multistep_order': 5           #idas/cvodes 2..5
    #         #'dump_in' : True,                  #(debug)
    #         #'dump_out' : True,                 #(debug)
    #     }
    # )
    # print("I (intg):", I )
    #
    # var1 = I( x0=[0, 0], p=0 )['xf']
    # var2 = I( x0=[0, 0], p=0.0215443 )['xf']
    # print("I ( evaluation at 'x0=[0,0,0,0], p=0' ):", var1 )
    # print("I ( evaluation at 'x0=[0,0,0,0], p=0.0215443' ):", var2 )
    # print("var1 - va2: ", var1 - var2)
    #
    #
    # I_accum = I.mapaccum(
    #     5,
    #     {
    #         #'verbose': True,
    #         'print_time' : True
    #     }
    # )
    # acum1 = I_accum( x0=[0,0], p=0 )['xf']
    # acum2 = I_accum( x0=[0,0], p=0.0215443 )['xf']
    # print('I_accum:', I_accum )
    # print("I_accum evaluation at '( x0=[0,0], p=0 )':", acum1 )
    # print("I_accum evaluation at '( x0=[0,0], p=0.0215443 )':", acum2 )
    # print("acum1-acum2: ",acum1 - acum2 )
    #
    #
    # x0Var = MX.sym('x0Var', Pendulum_using_RBDL_as_ODE.GetX().shape)
    # pVar = MX.sym('pVar', Pendulum_using_RBDL_as_ODE.GetP().shape)
    #
    # I_exp = I( x0=x0Var, p=pVar )
    # print("I_exp:", I_exp )
    #
    # I_exp_jac = jacobian( I_exp['xf'],
    #                       vertcat(
    #                           x0Var,
    #                           pVar
    #                       )
    #                     )
    # print("I_exp_jac:", I_exp_jac )
    #
    # F_I_exp_jac = Function(
    #     'F_I_exp_jac',
    #     [
    #         x0Var,
    #         pVar
    #     ],
    #     [ I_exp_jac ],
    #     ['x0','p'],
    #     ['dxf_dx0dp'],
    #     {
    #         #'verbose': True,
    #         'print_time' : True
    #     }
    # )
    # print("F_I_exp_jac:", F_I_exp_jac )
    #
    # print("F_I_exp_jac evaluation at '( [0, 0], 0 )':", F_I_exp_jac( [0, 0], 0 ) )
    #
    # hess = jacobian( jacobian( I_exp['xf'], vertcat(x0Var, pVar)), vertcat(x0Var, pVar) )
    # F_I_exp_hessian = Function(
    #     'F_I_exp_hessian',
    #     [ x0Var, pVar],
    #     [ hess ],
    #     ['x0', 'p'],
    #     ['dxf^{2}_dx0dp_{i}_dx0dp_{j}'],
    #     {
    #         #'verbose': True,
    #         'print_time' : True
    #     }
    # )
    # print("Hessian at '( [0, 0], 0 )':", F_I_exp_hessian(x0=[0, 0], p=0)['dxf^{2}_dx0dp_{i}_dx0dp_{j}'])

################### simbolical multibody dynamics using Feathearstone ###################

    print("\n###################\n Simbolical multibody dynamics using Feathearstone\n###################\n")

    PendulumCart_using_SymFederstone_as_ODE = \
        ClassDefinePendulumCart_Using_SymFederstone_as_ODE("PendulumCart_using_SymFederstone_as_ODE")

    print("ODE: ", PendulumCart_using_SymFederstone_as_ODE )
    print("xdot: ", PendulumCart_using_SymFederstone_as_ODE.xdot )
    xdot = Function('xdot',
                    [PendulumCart_using_SymFederstone_as_ODE.x, PendulumCart_using_SymFederstone_as_ODE.p],
                    [PendulumCart_using_SymFederstone_as_ODE.xdot],
                    ['x0','p'],
                    ['xdot']
                    )
    print(xdot)


    I = integrator(
        'I',
        'cvodes', #cvodes/idas/rk/collocation
        PendulumCart_using_SymFederstone_as_ODE.dae,
        {
            't0': 0, 'tf': 1,
            #'grid': [ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ],
            'enable_fd' : True,
            'enable_forward' : False, 'enable_reverse' : False, 'enable_jacobian' : False,
            #'fd_method' : 'smoothing',
            #'print_in':True,                   #print input value parameters(debug)
            #'print_out':True,                  #print output value parameters(debug)
            #'show_eval_warnings':True,         #(debug)
            #'inputs_check':True,               #(debug)
            #'regularity_check':True,           #(debug)
            #'verbose':True,                    #(debug)
            #'print_time' : True,               #(debug)
            #'simplify':True, 'expand':True,
            #'number_of_finite_elements': 5     # rk
            #'max_order' : 4                    #idas/cvodes 2/3/4/5
            #'max_multistep_order': 5           #idas/cvodes 2..5
            #'dump_in' : True,                  #(debug)
            #'dump_out' : True,                 #(debug)
        }
    )
    print("I (intg):", I )

    var1 = I( x0=[0, 0, 0, 0], p=0 )['xf']
    var2 = I( x0=[0, 0, 0, 0], p=0.0215443 )['xf']
    print("I ( evaluation at 'x0=[0,0,0,0], p=0' ):", var1 )
    print("I ( evaluation at 'x0=[0,0,0,0], p=0.0215443' ):", var2 )
    print("var1 - va2: ", var1 - var2)


    I_accum = I.mapaccum(
        5,
        {
            #'verbose': True,
            'print_time' : True
        }
    )
    acum1 = I_accum( x0=[0, 0, 0, 0], p=0 )['xf']
    acum2 = I_accum( x0=[0, 0, 0, 0], p=0.0215443 )['xf']
    print('I_accum:', I_accum )
    print("I_accum evaluation at '( x0=[0,0,0,0], p=0 )':", acum1 )
    print("I_accum evaluation at '( x0=[0,0,0,0], p=0.0215443 )':", acum2 )
    print("acum1-acum2: ",acum1 - acum2 )

    I2 = PendulumCart_using_SymFederstone_as_ODE.GetI(0,10)
    print(I2)
