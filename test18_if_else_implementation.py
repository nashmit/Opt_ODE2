import matplotlib.pyplot as plt

from casadi import *
import numpy as np

B = 1.5 # car width

h1 = 1.1 * B + 0.25
h2 = 3.5
h3 = 1.2 * B + 3.75
h4 = 1.3 * B + 0.25

x = MX.sym('x',1)

P_l = if_else( x <= 44, 0,
               if_else( x <= 44.5, 4 * h2 * ( x - 44 )**3,
                        if_else( x <= 45, 4 * h2 * ( x - 45 )**3 + h2,
                                 if_else( x <= 70, h2,
                                          if_else( x <= 70.5, 4 * h2 * ( 70 - x )**3 + h2,
                                                  if_else( x <= 71, 4 * h2 * ( 71 - x )**3, 0 ))))))

P_l_Function = Function('P_l_Function', [x], [P_l], ['x'], ['y'] )


P_u = if_else( x <= 15, h1,
               if_else( x <= 15.5, 4 * ( h3 - h1 ) * ( x - 15 )**3 + h1,
                       if_else( x <= 16, 4 * ( h3 - h1 ) * ( x - 16 )**3 + h3,
                                if_else( x <= 94, h3,
                                         if_else( x <= 94.5, 4 * ( h3 - h4 ) * ( 94 - x )**3 + h3,
                                                 if_else( x <= 95, 4 * ( h3 - h4 ) * ( 95 - x )**3 + h4, h4 ))))))

P_u_Function = Function('P_u_Function', [x], [P_u], ['x'], ['y'] )


#N = 100
#X = np.linspace(-50, 150, N )

#P_l_N = P_l_Function.map( N )
#Y = P_l_N(X)
#Y = Y.full()[0,:].T
#plt.plot(X, Y)

#P_u_N = P_u_Function.map( N )
#Y = P_u_N(X)
#Y = Y.full()[0,:].T
#plt.plot(X, Y)
##plt.axis('equal')
#plt.show()

#symvar(vertcat( 0, P_l_Function( x[0] ), 0.2, 0, 0, 0, 0 ))
#Function('SetLB_State', [symvar(vertcat( 0, P_l_Function( x[0] ), 0.2, 0, 0, 0, 0 ))], [ vertcat( 0, P_l_Function( x[0] ), 0.2, 0, 0, 0, 0 ) ] )

#x_in = SX.sym('x_in', 7)

#Function('SetLB_State', symvar(x_in), [ vertcat( 0, P_l_Function( x[0] ), 0.2, 0, 0, 0, 0 ) ] )

#func = lambda input : vertcat( 0, P_l_Function(input), 0.2, 0, 0, 0, 0 )
#print(func(46))