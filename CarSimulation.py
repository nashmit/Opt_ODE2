import Box2D
from Box2D.examples.framework import (Framework, main, Keys)
from Box2D.examples.backends.pygame_framework import PygameDraw

class CarSimulation(Framework):
    """You can use this class as an outline for your tests."""
    name = "CarSimulation"  # Name of the class to display
    description = "Car dynamics test!"

    def __init__(self):
        """
        Initialize all of your objects here.
        Be sure to call the Framework's initializer first.
        """
        super(CarSimulation, self).__init__()

        # Initialize all of the objects
        #self.settings.drawMenu = not self.settings.drawMenu
        #self.settings.pause = False

        self.viewCenter = (0, 0)


    def Keyboard(self, key):
        """
        The key is from Keys.K_*
        (e.g., if key == Keys.K_z: ... )
        """
        print('->' + str(key))

        pass

    def MouseDown(self, p):
        #print(p)
        pass

    def MouseUp(self, p):
        #print(p)
        pass

    def MouseMove(self, p):
        #print(p)
        pass

    def KeyboardUp(self, key):
        print(key)
        pass

    def Step(self, settings):
        """Called upon every step.
        You should always call
         -> super(Your_Test_Class, self).Step(settings)
        at the beginning or end of your function.

        If placed at the beginning, it will cause the actual physics step to happen first.
        If placed at the end, it will cause the physics step to happen after your code.
        """

        super(CarSimulation, self).Step(settings)

        # do stuff

        # Placed after the physics step, it will draw on top of physics objects
        #self.Print("*** Base your own testbeds on me! ***")

        #self.ConvertScreenToWorld(0,0)
        self.renderer.DrawCircle(center=self.renderer.to_screen( (0,0) ),
                                 radius=5, color=Box2D.b2Color(0.4, 0.9, 0.4))

        self.renderer.DrawSolidCircle(center=self.renderer.to_screen( (0,0) ), radius=2,
                                      axis= (0,0) ,
                                      color= Box2D.b2Color(0.4, 0.9, 0.4) )

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
