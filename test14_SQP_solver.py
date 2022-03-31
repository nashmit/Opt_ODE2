from casadi import *
import numpy as np

x = SX.sym('x',1)
y = SX.sym('y',1)
#X = vertcat(x,y)

# https://www.google.com/search?q=plot+-x*y*x&client=firefox-b-d&sxsrf=AOaemvK96RgJotErP4005N6H6zQTZoZL0Q%3A1641309648988&ei=0GXUYcv5O8qP9u8P3JS0mA0&ved=0ahUKEwjLoeD0spj1AhXKh_0HHVwKDdMQ4dUDCA0&uact=5&oq=plot+-x*y*x&gs_lcp=Cgdnd3Mtd2l6EAM6BwgAEEcQsANKBQg8EgExSgQIQRgASgQIRhgAUOEIWOkOYK8UaAFwAngAgAFkiAGvAZIBAzEuMZgBAKABAcgBBcABAQ&sclient=gws-wiz
Config= {
    'x':vertcat(x,y),
    'f':-x*y*x,
    'g':vertcat(y,x),
    #'lbx': [3,-2],#[-2, 3],
    #'ubx': [3, 2],#[ 2, 3],
    'lbg': [-3,-2],
    'ubg': [3, 2],
    'InitialGuess': [-3,1]#[3,-1]   #[ -1, 3 ] # [ 1, 3]
}

#A = DM.rand( Config['x'].numel(), Config['x'].numel() )
A = DM.eye(Config['x'].numel())
psd = 1/2 * (A + A.T)

H = psd
Grad = gradient( Config['f'], Config['x'] )

DeltaX = SX.sym('DeltaX', Config['x'].numel(), 1)

#inf >= lbg >= 0
lbg_ieq_exp = Config['g'] - Config['lbg']
#inf >= ubg >= 0
ubg_ieq_exp = Config['ubg'] - Config['g']
ieq_exp = vertcat(lbg_ieq_exp, ubg_ieq_exp)

lbg = vertcat( DM.zeros( lbg_ieq_exp.size1() ), DM.zeros( ubg_ieq_exp.size1() ) )
ubg = vertcat( DM.inf( lbg_ieq_exp.size1() ), DM.inf( ubg_ieq_exp.size1() ) )

#first order approximation
#vars = symvar(ieq_exp)
#g_FOA.call()
#vars = symvar( Config['x'] )
#cost_exp_eval = Function( 'cost_exp_eval', vars, [cost_exp] )
#cost = cost_exp_eval.call
#linearized_const = linearize( ieq_exp, Config['x'], Config['InitialGuess'] )
#linearized_const = linearize( ieq_exp, Config['x'],  )
#vars = symvar(Grad)
#evalf(substitute(substitute(Grad, vars[0] ,[1,1]),vars[1],[1,1]))



for indx in range(0,10):

    cost_exp = 1/2 * DeltaX.T @ H @ DeltaX + Grad.T @ DeltaX
    cost = substitute( cost_exp, Config['x'], DM(Config['InitialGuess']) )

    linearized_constraints = ieq_exp + jacobian( ieq_exp, Config['x'] ) @ DeltaX
    constraints = substitute(linearized_constraints, Config['x'], DM(Config['InitialGuess']) )

    qp = { 'x': DeltaX, 'f': cost, 'g': constraints }
    S = qpsol('S', 'qpoases', qp)
    print(S)
    if 'lam_x' in Config:
        sol = S(x0=Config['InitialGuess'], lbg=lbg, ubg=ubg, lam_x0=Config['lam_x'], lam_g0=Config['lam_g'])
    else:
        sol = S(x0=Config['InitialGuess'], lbg=lbg, ubg=ubg)

    Config['InitialGuess'] = Config['InitialGuess'] + 0.9*sol['x']

    Config['lam_x'] = sol['lam_x']
    Config['lam_g'] = sol['lam_g']

    #Lagrangian  = Config['f'] - Config['lam_g'] * ieq_exp
    #H = jacobian( jacobian(Lagrangian, Config['x']), Config['x'] )
    print(sol)
    print(Config['InitialGuess'])

