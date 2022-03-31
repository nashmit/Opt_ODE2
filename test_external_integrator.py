import Box2D
from Box2D.examples.framework import (Framework, main, Keys)
from Box2D.examples.backends.pygame_framework import PygameDraw
import pygame
from time import time

from copy import deepcopy
import numpy as np


from casadi import *

from DAE import DAEInterface

from Box2D import (b2PolygonShape, b2World)



world = b2World()  # default gravity is (0,-10) and doSleep is True
groundBody = world.CreateStaticBody(position=(0, -10),
                                    shapes=b2PolygonShape(box=(50, 10)),
                                    )

# Nice implementation / integration for pytorch:
# https://groups.google.com/g/casadi-users/c/gLJNzajFM6w/m/wBCNzzm7AQAJ
class XDot(Callback):
    def __init__(self, name, world, opts={}):
        Callback.__init__(self)
        self.world = world
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 3 # number of parameters
    def get_n_out(self): return 1 # size of state space

    def get_name_in(self, indx):
        switcher = {
            0:'p',
            1:'x',
            2:'y'
        }
        return switcher.get(indx,'wrong idx nr')

    def get_sparsity_in(self, indx):
        switcher = {
            0:Sparsity.dense(2),
            1:Sparsity.dense(1),
            2:Sparsity.dense(1)
        }

        return  switcher.get(indx)

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        print('bla')
        print(arg)
        return [ arg[2] ] # output must be return as list e.g.: "[ ]", same as in a regular Casadi function!

    #def eval(self, p,x,y):
        #y = arg[0].full()[0,0]

        ## Create a dynamic body at (0, 4)
        #body = self.world.CreateDynamicBody(position=(0, y))

        #return [body.position[1]]
        #return [arg[0] + arg[1]]
        #return [x+p]
        pass

xdot = XDot('xdot', world=world)
x = MX.sym('x',1)
y = MX.sym('y',1)

p = MX.sym('p',2)
#inp = vertcat(x,y,p)
#pp = xdot.call([x,y,p])
#print(xdot(DM([1,2]),DM([5,6])))
#print(pp)
#print(xdot.call([1,2,5]))
#print(xdot(1,2,5))
print(xdot)
xdot( p=vertcat(1,2), x=2, y=5 )
exp_xdot = xdot(p,x,y)
print( exp_xdot )
f_test = Function('f_test', [ x, y, p ], [ exp_xdot ], ['x','y','p'],['out'] )
print(f_test)



class ClassDefineBallInBox2D(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):

        world = b2World()  # default gravity is (0,-10) and doSleep is True
        groundBody = world.CreateStaticBody(position=(0, -10),
                                            shapes=b2PolygonShape(box=(50, 10)),
                                            )

        # Create a dynamic body at (0, 4)
        body = world.CreateDynamicBody(position=(0, 4))

        # And add a box fixture onto it (with a nonzero density, so it will move)
        box = body.CreatePolygonFixture(box=(1, 1), density=1, friction=0.3)

        ff = lambda state: Function(
            'lambda_lb_state', [ state ],
            [ vertcat( 0, P_l_Function( state[0] ), 3, 0, 0, 0, 0 ) ], ['current_state'], ['lb_state'] )


        # Prepare for simulation. Typically we use a time step of 1/60 of a second
        # (60Hz) and 6 velocity/2 position iterations. This provides a high quality
        # simulation in most game scenarios.
        timeStep = 1.0 / 60
        vel_iters, pos_iters = 6, 2


        # This is our little game loop.
        for i in range(60):
            # Instruct the world to perform a single step of simulation. It is
            # generally best to keep the time step and iterations fixed.
            world.Step(timeStep, vel_iters, pos_iters)

            # Clear applied body forces. We didn't apply any forces, but you should
            # know about this function.
            world.ClearForces()

            # Now print the position and angle of the body.
            print('iteration: ', i, ' time: ', (i+1)*timeStep,' ', body.position, body.angle)



        ##s = SX.sym('s',1) #postion
        ##v = SX.sym('v',1) #velocity e.g. v = ds/dt

        ##x = vertcat(s, v) # state space

        ##g = SX.sym('g',1) # gravitational acceleration
        ##q = vertcat(g) # parameter, same for all intervals, it can have lb<ub in this ex. lb=ub
        ##acc = SX.sym('acc',1) # extra acceleration / control parameter
        ##w = vertcat(acc) # control ( will be 0 ) lb<=0<=ub ( extra acceleration )
        ##p = vertcat( w, q)

        ##xdot = vertcat(v,g)
        xdot = world
        ##ode = {'x': x, 'p': p, 'ode': xdot}
        ode = {'x': x, 'p': p, 'ode': xdot}

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.xdot = xdot
        self.dae = ode




# integrator('I', 'idas',self.dae , {'t0': 0, 'tf': 1.0 / NumberShootingNodes })
# I = myintegrator('I','euler', world_simulation(as dae), { 't0': 0, 'tf': t_final } )
#f = MyIntegrator('f', 0.5, {'enable_fd':True})
#print(f(2))

