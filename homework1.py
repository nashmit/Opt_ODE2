from casadi import *
import numpy as np


#creating the problem which has to be solved
x = MX.sym('x',2)
p = MX.sym('p',4)

ode_rhs = vertcat(p[0]*x[0] - p[1]*x[0]*x[1], p[2]*x[0]*x[1] - p[3]*x[1])
ode = {'x': x,'p': p, 'ode': ode_rhs}

def get_measurements(m, interval, x_start, p_start):
    #m: number of measurements
    meastimes = np.linspace(interval[0], interval[1], m)
    meas_values = MX(m,2)
    for ii in range(m):
        opts = {'tf': meastimes[ii]}
        intg = integrator('intg', 'cvodes', ode, opts)
        result = intg(x0=x_start, p=p_start)
        meas_values[ii,:] = result['xf']
    return meas_values, meastimes

def get_F1_F2(m, shooting_grid, meas_values,sn, s, p):
    numx = meas_values.shape[1]

    #computation of F1
    F1 = MX.sym('F1', numx * shooting_grid.shape[0])

    for jj in range(shooting_grid.shape[0]):
        F1[jj * numx:(jj + 1) * numx] = s[jj, :] - meas_values[jj,:]

    #computation of F2
    F2 = MX.sym('F2', numx * (shooting_grid.shape[0] - 1))
    for ii in range(shooting_grid.shape[0] - 1):
        opts = {'t0': shooting_grid[ii], 'tf': shooting_grid[ii + 1]}
        I = integrator('I', 'cvodes', ode, opts)

        res = I(x0=s[ii], p=p)
        xf_exp = res['xf']

        xf_fun = Function('xf_fun', [s], [xf_exp])

        F2[ii * numx:(ii + 1) * numx] = xf_fun(sn[ii]) - sn[ii+1]

    return F1, F2

def get_J1_J2(F1, F2, sn, s, p):

    J1_exp = jacobian(F1, s)
    J1_exp2 = jacobian(F1, p)
    J1_func = Function('J1_func', [s], [J1_exp])
    J1_func2 = Function('J1_func2', [p], [J1_exp2])
    J1_s = J1_func(s)
    J1_p = J1_func2(p)
    J1 = horzcat(J1_s, J1_p)


    J2_exp = jacobian(F2, s)
    J2_exp2 = jacobian(F2, p)
    J2_func = Function('J2_func', [s], [J2_exp])
    J2_func2 = Function('J2_func2', [p], [J2_exp2])
    J2_s = J2_func(s)
    J2_p = J2_func2(p)
    J2 = horzcat(J2_s, J2_p)

    return J1, J2


def constrained_gauss_newton(f, x0, itmax=100, tol=1e-7, verbose=1):
    x = np.copy(x0)

    for ii in range(itmax):
        F1, F2, J1, J2 = f(x)


        #computation of dx (solving the quadratic program)
        dx = MX.sym('dx', J1.shape[1])
        qp = {'x':dx, 'f':0.5 * norm_2(F1 + J1@dx)**2, 'g': F2 + J2@dx}
        dx = qpsol('dx', 'qpoases', qp)



        # print informations
        norm_dx = np.linalg.norm(dx)
        if verbose > 0:
            print("it: {:3d}, \u2016F1\u2016: {:e}, \u2016F2\u2016: {:2.1e}, \u2016dx\u2016: {:2.1e}".format(
                ii, np.linalg.norm(F1), np.linalg.norm(F2), norm_dx), end="")
        if np.linalg.norm(dx) < tol:
            if verbose > 0:
                print("")  # empty line for a nicer output
            break

        x += dx

    return x


#True start value x_start and p_start
x0 = np.array([20., 10.])
p0 = np.array([.2, .01, .001, .1])

number_measurements = 10

result, shooting_grid = get_measurements(number_measurements, np.array([0,10]), x0, p0)

noise = 1
meas_values = result + noise*np.random.rand(number_measurements,2)

def F_J(sp):
    F1, F2 = get_F1_F2(number_measurements, shooting_grid, meas_values, sn, s, p0)
    J1, J2 = get_J1_J2(F1, F2, sn, s, p)

    return F1, F2, J1, J2


sn = MX.sym('sn', number_measurements)
s = MX.sym('s', number_measurements)

F1, F2 = get_F1_F2(number_measurements, shooting_grid, meas_values, sn, s, p0)
print(F1, F2)

J1 = get_J1_J2(F1, F2, sn, s, p)
print('J1', J1)

sp = np.concatenate([x0, p0])
sp = constrained_gauss_newton(F_J, sp)


I = integrator('I','cvodes', ode)
res = I(x0=x0,p=p0)
xf_exp = res['xf']

print(res['xf'])


