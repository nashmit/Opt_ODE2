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



class XDot(Callback):

    def __init__(self, name, world, opts ):
        Callback.__init__(self)
        self.world = world
        self.construct( name, opts = {'enable_fd':True} )

    # Number of inputs and outputs
    def get_n_in(self):
        return 2 # number of parameters
    def get_n_out(self):
        return 1 # size of state space

    def get_name_in(self, indx):
        switcher = {
            0:'x',
            1:'p',
        }
        return switcher.get(indx, 'wrong idx nr' )
    def get_name_out(self, indx):
        switcher = {
            0:'Simulation',
        }
        return switcher.get(indx, 'wrong idx nr' )

    def get_sparsity_in(self, indx):
        switcher = {
            0:Sparsity.dense(2),
            1:Sparsity.dense(1),
        }
        return switcher.get(indx)
    def get_sparsity_out(self, indx):
        switcher = {
            0:Sparsity.dense(1),
        }
        return switcher.get(indx)

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):

        #xdot = vertcat(
        #    thetaDot,
        #    -g / l * sin( theta ) + acc
        #)

        print('inside eval:')
        print('eval arg:', arg)
        # output must be return as list e.g.: "[ ]", same as in a regular Casadi function!
        return [ arg[2] ]
        pass


xdot = XDot('xdot',None,{})