from Box2D import *
from Box2D.examples.framework import (Framework, main, Keys)
from Box2D.examples.backends.pygame_framework import PygameDraw

import pygame

def Box2d_build_test(world=None):

    if world==None:
        world = b2World()

    world.gravity = b2Vec2( 0, 0 )

    PENDULUM_LENGTH = 10
    #pendulum_shape = b2PolygonShape( box=(0.5, 0.5 * PENDULUM_LENGTH))
    pendulum_shape = b2PolygonShape( vertices=[ (-1,1), (1,1),
                                                (1,-1 * PENDULUM_LENGTH),
                                                (-1,-1 * PENDULUM_LENGTH) ] )
    pendulum_bd = b2BodyDef( type=b2_dynamicBody,
                             #position=(0, 12 - 0.5 * PENDULUM_LENGTH), angle=-3.14/2, allowSleep=False )
                             position=(0, 0), angle=0, allowSleep=False )
    pendulumBody = world.CreateBody(pendulum_bd)
    pendulumBody.CreateFixture(shape=pendulum_shape, density=1)


    return world, pendulumBody

pass

def Box2d_world_Init_cart_inverted_pendulum(world=None):

    if world==None:
        world = b2World()

    #[P]: gravity
    world.gravity = b2Vec2( 0, -10 )

    #ground
    ground = world.CreateStaticBody(allowSleep=False)
    edgeShape = b2EdgeShape()
    edgeShape.vertices = [ (-20.0, 0.0), (20.0, 0.0) ]
    fd = b2FixtureDef(shape=edgeShape, density=1, friction=1)
    ground.CreateFixture(fd)

    #cart
    #[P] cart_w, cart_h, cart_density
    box_shape = b2PolygonShape( box=(2,1) )
    box_bd = b2BodyDef( type=b2_dynamicBody, position=(0,12), allowSleep=False )
    cartBody = world.CreateBody(box_bd)
    cartBody.CreateFixture(shape=box_shape, density=1, categoryBits=0 )

    #joint ground-cart
    jd = b2PrismaticJointDef( bodyA=ground, bodyB=cartBody, localAnchorA=(0,12), localAnchorB=(0,0),)
                              #motorSpeed=0, maxMotorForce=1000, enableMotor=True )
    prismaticJoint = world.CreateJoint(jd)

    #pendulum
    # [P] Pendulum_w, Pendulum_length, Pendulum_density
    PENDULUM_LENGTH = 10
    pendulum_shape = b2PolygonShape( box=(0.5, 0.5 * PENDULUM_LENGTH))
    pendulum_bd = b2BodyDef( type=b2_dynamicBody, position=(0, 12 - 0.5 * PENDULUM_LENGTH), allowSleep=False )
    pendulumBody = world.CreateBody(pendulum_bd)
    pendulumBody.CreateFixture(shape=pendulum_shape, density=1)

    jd2 = b2RevoluteJointDef( bodyA=cartBody, bodyB=pendulumBody,
                              localAnchorA=(0,0), localAnchorB=(0, 0.5 * PENDULUM_LENGTH) )
    pendulumJoint = world.CreateJoint(jd2)

    #proxy obj used in setting/getting theta and thetaDot
    proxy_orientation_obj = world.CreateBody(box_bd)
    #proxy_orientation_obj.CreateFixture(shape=b2PolygonShape(box=(0.5,0.5)), density=1,categoryBits=0)
    proxy_orientation_obj.CreateFixture(shape=b2CircleShape(radius=0.1), density=1,categoryBits=0)

    jd3 = world.CreateWeldJoint( bodyA=proxy_orientation_obj, bodyB=pendulumBody,
                                 anchor=proxy_orientation_obj.worldCenter )

    return world, cartBody, proxy_orientation_obj
    pass

def Test():
    world = Box2d_world_Init_cart_inverted_pendulum()

    timeStep = 1.0 / 60
    vel_iters, pos_iters = 6, 2

    for i in range(60):
        # Instruct the world to perform a single step of simulation. It is
        # generally best to keep the time step and iterations fixed.
        world.Step(timeStep, vel_iters, pos_iters)

        # Clear applied body forces. We didn't apply any forces, but you should
        # know about this function.
        world.ClearForces()

        # Now print the position and angle of the body.
        #print(body.position, body.angle)
        print('{')
        for indx in range(0,len(world.bodies)):
            print('\t',world.bodies[indx].position, world.bodies[indx].angle)
        print('}')
#Test()


class Test(Framework):
    """You can use this class as an outline for your tests."""
    name = "Test"  # Name of the class to display
    description = "The description text goes here"

    def __init__(self):
        """
        Initialize all of your objects here.
        Be sure to call the Framework's initializer first.
        """
        super(Test, self).__init__()

        self.viewCenter = (0, 0)

        #test 1
        self.world = Box2d_world_Init_cart_inverted_pendulum(self.world)[0]
        #test 2
        #self.world = Box2d_build_test(self.world)[0]

        # Initialize all of the objects

        print(self.world)

    def Keyboard(self, key):
        """
        The key is from Keys.K_*
        (e.g., if key == Keys.K_z: ... )
        """
        pass

    def Step(self, settings):
        """Called upon every step.
        You should always call
         -> super(Your_Test_Class, self).Step(settings)
        at the beginning or end of your function.

        If placed at the beginning, it will cause the actual physics step to happen first.
        If placed at the end, it will cause the physics step to happen after your code.
        """

        super(Test, self).Step(settings)

        # do stuff

        # Placed after the physics step, it will draw on top of physics objects
        self.Print("*** Base your own testbeds on me! ***")

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
    main(Test)