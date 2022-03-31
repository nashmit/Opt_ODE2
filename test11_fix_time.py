from casadi import *

t0 = 0
tf = 2

x0 = SX.sym('x0');
x1 = SX.sym('x1');
x = vertcat(x0, x1);
p00 = SX.sym('p00');
p01 = SX.sym('p01');
p10 = SX.sym('p10');
p11 = SX.sym('p11');
p = vertcat(horzcat(p00, p01), horzcat(p10, p11));
xdot = vertcat(p00 * x0 - p01 * x0 * x1, p10 * x0 * x1 - p11 * x1);
ode = {'x': x, 'p': reshape(p, 4, 1), 'ode': xdot};
I = integrator('I', 'cvodes', ode, {'t0': t0, 'tf': tf});
print( I(x0 = [20,10], p=[0.2, 0.01, 0.001, 0.1])['xf'] )


#rescale time interval from [0,2] to [0,1] and
#intruduce the scale factor as a parameter for xdot by scalling the ODE
t0 = 0
tf = 1

x0 = SX.sym('x0');
x1 = SX.sym('x1');
x = vertcat(x0, x1);
p00 = SX.sym('p00');
p01 = SX.sym('p01');
p10 = SX.sym('p10');
p11 = SX.sym('p11');
p = vertcat(horzcat(p00, p01), horzcat(p10, p11));
xdot = vertcat(p00 * x0 - p01 * x0 * x1, p10 * x0 * x1 - p11 * x1);

#time rescale so that we end with the same result
xdot = xdot * (2-0)

ode = {'x': x, 'p': reshape(p, 4, 1), 'ode': xdot};
I = integrator('I', 'cvodes', ode, {'t0': t0, 'tf': tf});
print( I(x0 = [20,10], p=[0.2, 0.01, 0.001, 0.1])['xf'] )