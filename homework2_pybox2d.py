import Box2D
from Box2D.examples.framework import (Framework, main, Keys)
from Box2D.examples.backends.pygame_framework import PygameDraw

from DAE import *
from copy import deepcopy

import numpy as np
import pygame

from time import time

from homework2 import OCP, OptSolver

class Car:

    def __init__(self, world, renderer ):

        self.renderer = renderer

        self.Car_ODE = ClassDefineOneShootingNodForCar("Car")
        self.GroundTruthCar = deepcopy( Car_Conf2 )

        self.world = world
        self.car_width = self.GroundTruthCar['car_width'] # meters

        #car dimensions
        self.l_r = self.GroundTruthCar['q'][ 3 ]
        self.l_f = self.GroundTruthCar['q'][ 2 ]

        #position
        c_x = self.GroundTruthCar['Sn'][ 0 ]
        c_y = self.GroundTruthCar['Sn'][ 1 ]
        self.c_x:float = c_x.full()[0,0]
        self.c_y:float = c_y.full()[0,0]

        # global orientation in rad
        psi = self.GroundTruthCar['Sn'][ 5 ]
        self.psi:float = psi.full()[0,0]

        self.body = self.world.CreateStaticBody(
            shapes = Box2D.b2PolygonShape( vertices = [
                ( -self.l_r, self.car_width / 2 ),
                (  self.l_f, self.car_width / 2 ),
                (  self.l_f,-self.car_width / 2 ),
                ( -self.l_r,-self.car_width / 2 ) ] ),
            position = ( self.c_x, self.c_y ),
            angle = self.psi
        )

        wheel_x_len = (self.l_r + self.l_f) / 3
        wheel_y_len = (self.car_width) / 3

        # Small box
        box = Box2D.b2FixtureDef(
            shape=Box2D.b2PolygonShape(vertices = [
                ( -wheel_x_len/2, wheel_y_len / 2 ),
                (  wheel_x_len/2, wheel_y_len / 2 ),
                (  wheel_x_len/2,-wheel_y_len / 2 ),
                ( -wheel_x_len/2,-wheel_y_len / 2 ) ]),
            density=1,
            #restitution=0,
        )

        self.front_wheel = self.world.CreateDynamicBody(
            position = ( self.c_x + self.l_f + wheel_x_len, self.c_y + 0),
            fixtures=box,
        )

        #self.front_wheel = self.world.CreateDynamicBody(
        #    shapes = Box2D.b2PolygonShape( vertices = [
        #        ( -wheel_x_len/2, wheel_y_len / 2 ),
        #        (  wheel_x_len/2, wheel_y_len / 2 ),
        #        (  wheel_x_len/2,-wheel_y_len / 2 ),
        #        ( -wheel_x_len/2,-wheel_y_len / 2 ) ] ),
        #    position = ( self.c_x + self.l_f + wheel_x_len, self.c_y + 0),
        #    angle = self.psi,
        #    #density=0.1
        #)

        #self.world.CreateWeldJoint(bodyA=self.body, bodyB=self.front_wheel, anchor=self.body.worldCenter)
        self.world.CreateRevoluteJoint(bodyA=self.body, bodyB=self.front_wheel, anchor=self.front_wheel.worldCenter)
        #self.world.CreatePrismaticJoint(
        #    bodyA=self.body,
        #    bodyB=self.front_wheel,
        #    anchor=self.body.worldCenter,
        #    axis=(1, 0),
        #    lowerTranslation=0,#-5.0,
        #    upperTranslation=0,#2.5,
        #    enableLimit=True,
        #    #motorForce=1.0,
        #    #motorSpeed=0.0,
        #    #enableMotor=True,
        #)

    #def Update(self, deltaTime):
    def Update(self):

        self.front_wheel.awake = True

        #I = self.Car_ODE.GetI( 0, deltaTime )
        #I = self.Car_ODE.GetI( 0, deltaTime )
        I = self.Car_ODE.GetI( self.GroundTruthCar['t0'], self.GroundTruthCar['tf'] )

        #t_step = time()
        self.GroundTruthCar['Sn'] = \
            I( x0 = self.GroundTruthCar['Sn'],
               p = self.GroundTruthCar['w'] + self.GroundTruthCar['q'] )['xf']
        #t_step = time() - t_step
        #print('t_step: '+str(t_step))

        #position
        c_x = self.GroundTruthCar['Sn'][ 0 ]
        c_y = self.GroundTruthCar['Sn'][ 1 ]
        self.c_x = c_x.full()[0,0]
        self.c_y = c_y.full()[0,0]

        # global orientation in rad
        psi = self.GroundTruthCar['Sn'][ 5 ]
        self.psi = psi.full()[0,0]

        #wheel angle
        delta = self.GroundTruthCar['Sn'][ 3 ]
        self.delta = delta.full()[0,0]


        self.body.position = ( self.c_x, self.c_y )
        self.body.angle = self.psi

        self.front_wheel.angle = self.psi + self.delta

        #print(self.body.position  )
        #print(self.body.angle )

        #print(self.GroundTruthCar['Sn'][2])

        pass

    def DrawLine(self, vertices, color):

        if not vertices:
            return

        pygame.draw.aalines(self.renderer.surface, color.bytes, False, vertices)

        pass

    def ComputeAndDrawTrajectoryWithoutUpdate(self):

        self.front_wheel.awake = True

        I_plot = self.Car_ODE.GetI_plot( self.GroundTruthCar['t0'], self.GroundTruthCar['tf'] )

        path = I_plot(
            x0 = self.GroundTruthCar['Sn'],
            p = self.GroundTruthCar['w'] + self.GroundTruthCar['q'] )['xf']

        path_x = path[0,:]
        path_y = path[1,:]
        path_x = path_x.full()[0,:]
        path_y = path_y.full()[0,:]
        path_aslist = list( zip( path_x, path_y ) )

        path_aslist = [self.renderer.to_screen( node ) for node in path_aslist]

        self.DrawLine( path_aslist, Box2D.b2Color(0.8, 1, 0.2) )

        pass

class EndPoint:

    def __init__( self, renderer,color ):
        self.nodes_X = []
        self.nodes_Y = []
        self.renderer = renderer
        self.color = color
        pass

    def AddPoint(self, x_pos, y_pos):
        self.nodes_X = [ x_pos ]
        self.nodes_Y = [ y_pos ]
        pass

    def PaintPoints(self):
        #self.ConvertScreenToWorld(0,0)

        for Point in zip( self.nodes_X, self.nodes_Y ):
            self.renderer.DrawSolidCircle(
                center=self.renderer.to_screen( Point ), radius=0.4,
                axis= (0,0) ,
                color= self.color )
        pass


class Spline:

    def __init__( self, renderer,color ):
        self.nodes_X = []
        self.nodes_Y = []
        self.renderer = renderer
        self.color = color
        pass

    def AddPoint(self, x_pos, y_pos):
        self.nodes_X.append( x_pos )
        self.nodes_Y.append( y_pos )
        pass

    def PaintPoints(self):
        #self.ConvertScreenToWorld(0,0)

        for Point in zip( self.nodes_X, self.nodes_Y ):
            self.renderer.DrawSolidCircle(
                center=self.renderer.to_screen( Point ), radius=0.4,
                axis= (0,0) ,
                color= Box2D.b2Color(0.4, 0.9, 0.4) )
        pass

    def DrawLine(self, vertices, color):

        if not vertices:
            return

        pygame.draw.aalines(self.renderer.surface, color.bytes, False, vertices)

        pass

    def PaintSpline(self):

        if len(self.nodes_X) <=5:
            return

        lut = interpolant('LUT','bspline',[self.nodes_X], self.nodes_Y)
        x = np.linspace(self.nodes_X[0], self.nodes_X[-1], 100)
        nodes = list(zip(x, lut(x).full()[:,0] ))

        nodes = [self.renderer.to_screen( node ) for node in nodes]

        self.DrawLine(nodes, self.color )

        pass

    def Paint(self):
        self.PaintPoints()
        self.PaintSpline()
        pass

    pass

class CarSimulation(Framework):
    """You can use this class as an outline for your tests."""
    name = "CarSimulation"  # Name of the class to display
    description = "Car test!"

    def __init__(self):
        """
        Initialize all of your objects here.
        Be sure to call the Framework's initializer first.
        """
        super(CarSimulation, self).__init__()

        # Initialize all of the objects
        #self.settings.drawMenu = not self.settings.drawMenu
        #self.settings.pause = False

        self.endPoint = EndPoint( self.renderer, Box2D.b2Color( 1, 0, 0) )


        self.RecordingInitialGuess = \
            {
                'Xf':[],
                'tf':0,
                'NumberShootingNodes':0,
                'Sn':[],
                'w':[],
            }

        self.UpdateCurrentProblmConf = 0

        self.viewCenter = (0, 0)
        
        self.world.gravity = Box2D.b2Vec2( 0, 0)

        self.Car = Car( self.world, self.renderer )

        self.LowerSpline = Spline( self.renderer,Box2D.b2Color( 0, 0.9, 0.4) )
        self.UpperSpline = Spline( self.renderer,Box2D.b2Color( 1, 0.9, 0.4) )

        self.renderer.axisScale = 3


    def Keyboard(self, key):
        """
        The key is from Keys.K_*
        (e.g., if key == Keys.K_z: ... )
        """
        #print('->' + str(key))

        pass

    def MouseDown(self, p):
        #print(p)
        pass

    def MouseUp(self, p):
        #print(p)
        if self.settings.BuildLowerSpline:
            self.LowerSpline.AddPoint( p.x, p.y )

        if self.settings.BuildUpperSpline:
            self.UpperSpline.AddPoint( p.x, p.y )

        self.point = (p.x, p.y)

        pass

    def MouseMove(self, p):
        #print(p)
        pass

    def KeyboardUp(self, key):
        #print(key)

        if key == Keys.K_l:
            self.settings.BuildLowerSpline = not self.settings.BuildLowerSpline
            self.gui_table.updateGUI(self.settings)

        if key == Keys.K_a:
            self.settings.BuildUpperSpline = not self.settings.BuildUpperSpline
            self.gui_table.updateGUI(self.settings)

        if key ==Keys.K_r:
            self.settings.RecordingEndPoint = not self.settings.RecordingEndPoint
            self.gui_table.updateGUI(self.settings)


        assert not (self.settings.BuildLowerSpline == True and self.settings.BuildUpperSpline == True), "You can't build 2 splines in the same time!"

    pass

    def Step(self, settings):
        """Called upon every step.
        You should always call
         -> super(Your_Test_Class, self).Step(settings)
        at the beginning or end of your function.

        If placed at the beginning, it will cause the actual physics step to happen first.
        If placed at the end, it will cause the physics step to happen after your code.
        """

        if settings.pause==True:
            self.Car.ComputeAndDrawTrajectoryWithoutUpdate()

            if self.settings.RecordingEndPoint:
                #self.settings.RecordingEndPoint = not self.settings.RecordingEndPoint
                if hasattr(self,"point"):
                    self.endPoint.AddPoint( self.point[0], self.point[1] )

        self.gui_table.updateSettings(self.settings)
        if settings.oneStep == True:
            settings.pause = False
            self.gui_table.updateGUI( settings )



        # do stuff

        # Placed after the physics step, it will draw on top of physics objects
        #self.Print("*** Base your own testbeds on me! ***")

        #self.ConvertScreenToWorld(0,0)

        #self.renderer.DrawCircle(center=self.renderer.to_screen( (0,0) ),
        #                         radius=5, color=Box2D.b2Color(0.4, 0.9, 0.4))

        #self.renderer.DrawSolidCircle(center=self.renderer.to_screen( (0,0) ), radius=2,
        #                              axis= (0,0) ,
        #                              color= Box2D.b2Color(0.4, 0.9, 0.4) )
        #self.renderer.DrawTransform( self.Car.body.transform )
        self.Print( 'Recording Initial Guess: ' + str(settings.RecordingInitialGuess), (0,255,0,0) )
        self.Print( 'Simulate Using Optimal Control Result: ' + str(settings.SimulateUsingOptimalControlResult),
                    (255,0,0,0) )
        self.Print( 'Delta time: ' + str(settings.DeltaTime) + ' ( seconds )' )
        self.Print( 'Total Time: ' + str(settings.TotalTime) + ' ( seconds )', (0,255,0,0) )

        self.Print( 'Omega_delta: ' + str(settings.omega_delta) + ' ( rad/sec  ->angular velocity )' )
        self.Print( 'F_B: ' + str(settings.F_B) + ' ( N ->Total braking force )' )
        self.Print( 'Phi: ' + str(settings.phi) + ' ( None: - acceleration pedal position)' )
        self.Print( '---')
        self.Print( str( self.Car.GroundTruthCar['SnName'] ) )
        self.Print( 'State spate: ' + str( self.Car.GroundTruthCar['Sn'] ) )

        self.Print("Current Problem Conf id update! " + str( self.UpdateCurrentProblmConf ), (255,0,0,0) )

        if len(self.endPoint.nodes_X)>0:
            self.Print('End Point: '+ str(self.endPoint.nodes_X[0]) + str(' ') + str(self.endPoint.nodes_Y[0]),
                       (255,0,0,0)  )


        if settings.SimulateUsingOptimalControlResult==False:
            self.Car.GroundTruthCar['w'] = [ settings.omega_delta, settings.F_B, settings.phi ]
            self.Car.GroundTruthCar['tf'] =  settings.DeltaTime #1/settings.hz

        #if settings.oneStep == True or settings.pause == False:
        if settings.pause == False:
            self.Car.Update()
            #self.Car.Update( 1/settings.hz )


        super(CarSimulation, self).Step(settings)


        self.LowerSpline.Paint()
        self.UpperSpline.Paint()
        self.endPoint.PaintPoints()

        if settings.oneStep == True:
            settings.TotalTime += settings.DeltaTime
            settings.oneStep = False
            settings.pause = True

            if settings.RecordingInitialGuess:
                # record current initial guess
                self.UpdateRecordingInitialGuess(settings)

            self.gui_table.updateGUI( settings )

        if settings.UpdateCurrentProblmConf == True:
            settings.UpdateCurrentProblmConf = False
            self.UpdateCurrentProblmConf +=1

            #apply optimal control using current initial guess and current conf

            self.ComputeOCP()

            self.gui_table.updateGUI( settings )

    def ComputeOCP(self):

        # copy CarConfig and apply changes to it: ( e.g. end time,
        Config = deepcopy(Car_Conf2)
        Config['tf'] = self.RecordingInitialGuess['tf']
        Config['tf_ub'] = Config['tf'] + 1 # + 5 # add extra time
        Config['NumberShootingNodes'] = self.RecordingInitialGuess['NumberShootingNodes']
        Sn = self.RecordingInitialGuess['Sn']
        Config['Sn'][0] = self.RecordingInitialGuess['Sn'][0] # first node
        Config['Xf'][0] = self.RecordingInitialGuess['Xf'][0] # overwrite  only 'x' position
        # construct parameter
        q = repmat( vertcat( DM(Config['q'] ), self.RecordingInitialGuess['tf'] ),
                1, self.RecordingInitialGuess['NumberShootingNodes'] )
        P = vertcat( self.RecordingInitialGuess['w'], q )

        lut_LowerSpline = interpolant('LUT_LowerSpline','bspline',
                                      [self.LowerSpline.nodes_X], self.LowerSpline.nodes_Y)
        lut_UpperSpline = interpolant('LUT_LowerSpline','bspline',
                                      [self.UpperSpline.nodes_X], self.UpperSpline.nodes_Y)

        # overwrite 'lb_state' and 'ub_state' lambda functions from 'GroundTruthCar' !!!
        P_l_Function = lut_LowerSpline
        P_u_Function = lut_UpperSpline

        Config['lb_state'] = \
            lambda state: Function(
                'lambda_lb_state', [ state ],
                [ vertcat( 0, P_l_Function( state[0] ), 0.2, 0, 0, 0, 0 ) ], ['current_state'], ['lb_state'] )

        Config['ub_state'] = \
            lambda state: Function(
                'lambda_ub_state',  [ state ],
                [ vertcat( 0, P_u_Function( state[0] ),   0, 0, 0, 0, 0 ) ], ['current_state'], ['ub_state'] )

        # set start and end position of the car
        # check what other states need to be updated too!



        tf = SX.sym('tf',1)
        Mayer_exp = tf
        Car_ODE = ClassDefineOneShootingNodForCar("Car")
        #sumsqr( Car_ODE.GetW() ) + sumsqr( Car_ODE.GetX() )
        ocp = OCP( "Optimize Simulated Car" ). \
            AddDAE( Car_ODE ). \
            AddLagrangeCostFunction( L = 1.0/Car_ODE.GetX()[2] ). \
            AddMayerCostFunction( M = Mayer_exp ). \
            SetStartTime( t0 = Config['t0'] ). \
            SetEndTime( tf = tf ). \
            SetX_0( x0 = Config['Sn'], mask=Config['S0_mask'] ). \
            SetX_f( xf = Config['Xf'], mask=Config['Xf_mask'] ). \
            SetLBW(lbw=Config['lbw'] ). \
            SetUBW(ubw=Config['ubw'] ). \
            SetLBQ(lbq=Config['lbq'] ). \
            SetUBQ(ubq=Config['ubq'] ). \
            SetLB_State( eq=Config['lb_state'] , mask=Config['lb_state_mask'] ). \
            SetUB_State( eq=Config['ub_state'] , mask=Config['ub_state_mask'] ). \
            SetNumberShootingNodes( Number = Config['NumberShootingNodes'] ). \
            SetSolver( Solver = OptSolver.nlp )
        #    Build()
        ocp.Build( Config, (Sn, P) )
        # self.result = ocp.Build( Config, (Sn, P) )

        pass


    def UpdateRecordingInitialGuess(self, settings):

        # update end state
        #update total time
        # update shooting nodes
        # add current state space in a vector
        # add current control vector

        #assert False, "Xf is not recorded!"
        if len(self.endPoint.nodes_X)>0:
            self.RecordingInitialGuess['Xf'] =[self.endPoint.nodes_X[0], self.endPoint.nodes_Y[0]]
        else:
            print("Warning: don't forget to set the end point!")
        self.RecordingInitialGuess['tf'] = settings.TotalTime
        self.RecordingInitialGuess['NumberShootingNodes'] +=1
        self.RecordingInitialGuess['Sn'] = \
            horzcat( self.RecordingInitialGuess['Sn'], self.Car.GroundTruthCar['Sn'] )
        ['omega_delta', 'F_B', 'phi']
        self.RecordingInitialGuess['w'] = \
            horzcat(
                self.RecordingInitialGuess['w'],
                vertcat( settings.omega_delta, settings.F_B, settings.phi ) )

        print(self.RecordingInitialGuess)
        pass

    def ShapeDestroyed(self, shape):
        """
        Callback indicating 'shape' has been destroyed.
        """
        pass

    def JointDestroyed(self, joint):
        """
        The joint passed in was removed.
        """
        pass

    # More functions can be changed to allow for contact monitoring and such.
    # See the other testbed examples for more information.

if __name__ == "__main__":
    main(CarSimulation)
