from casadi import *
import numpy as np

x = SX.sym('x',1)
y = SX.sym('y',1)
p = SX.sym('p',1) #parameter

#the order or the symbolical variables from "X" is used for upper bounds / lower bounds vector!!!
#also, the output is in the same order and the initial guss require the same order

# https://www.google.com/search?q=plot+-x*y*x&client=firefox-b-d&sxsrf=AOaemvK96RgJotErP4005N6H6zQTZoZL0Q%3A1641309648988&ei=0GXUYcv5O8qP9u8P3JS0mA0&ved=0ahUKEwjLoeD0spj1AhXKh_0HHVwKDdMQ4dUDCA0&uact=5&oq=plot+-x*y*x&gs_lcp=Cgdnd3Mtd2l6EAM6BwgAEEcQsANKBQg8EgExSgQIQRgASgQIRhgAUOEIWOkOYK8UaAFwAngAgAFkiAGvAZIBAzEuMZgBAKABAcgBBcABAQ&sclient=gws-wiz
Config= {
    'x':vertcat(x,y),
    'p':p,
    'f':-x*y*x*p,
    'g':vertcat(y,x),
    #'lbx': [3,-2],#[-2, 3],
    #'ubx': [3, 2],#[ 2, 3],
    'lbg': [3,-2],
    'ubg': [3, 2],
    'InitialGuess': [3,1], #[3,-1]   #[ -1, 3 ] # [ 1, 3]
    'InitialGuess_P': 1,
}

nlp = {'x': Config['x'],
       'p':Config['p'],
       'f': Config['f'],
       'g': Config['g']
       }

SolverNLP = nlpsol('S', 'ipopt', nlp, {})
print(SolverNLP)
#SolverNLP( x0 = vertcat(x,y) , p = p ) # Not yet defined / implemented !?
#jacobian(SolverNLP,p)

result = SolverNLP(
    x0 = Config['InitialGuess'],
    p = Config['InitialGuess_P'],
    #lbx=Config['lbx'],
    #ubx=Config['ubx'],
    lbg = Config['lbg'],
    ubg = Config['ubg']
)

print(result)