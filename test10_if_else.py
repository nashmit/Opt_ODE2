from casadi import *

#maximum or if a>b return a
#(a+b + abs(a-b))/2

a = SX.sym('a')
b = SX.sym('b')

c = (a + b + fabs(a-b))/2
print(c)
print(jacobian(c,vertcat(a,b)))
print(gradient(c,vertcat(a,b)))
print('--')
print(jacobian(if_else(a>b,a,b),vertcat(a,b)) )
print(gradient(if_else(a>b,a,b),vertcat(a,b)) )
print('--')
print(gradient(fabs(a-b),vertcat(a,b)))
print('--')
print(jacobian(fabs(a),vertcat(a,b)))
print('--')
print(gradient(sign(a-b),vertcat(a,b)))
