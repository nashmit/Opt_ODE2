from casadi import *

class MyCallback(Callback):
    def __init__(self, name, nx, ng, np, opts={}):
        Callback.__init__(self)

        self.nx = nx
        self.ng = ng
        self.np = np

        # Initialize internal objects
        self.construct(name, opts)

    def get_n_in(self): return nlpsol_n_out()
    def get_n_out(self): return 1
    def get_name_in(self, i): return nlpsol_out(i)
    def get_name_out(self, i): return "ret"

    def get_sparsity_in(self, i):
        n = nlpsol_out(i)
        if n=='f':
            return Sparsity. scalar()
        elif n in ('x', 'lam_x'):
            return Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return Sparsity.dense(self.ng)
        else:
            return Sparsity(0,0)

    def eval(self, arg):
        return [0]

class Optimization(object):
    def __init__(self):

        # Rosenbrock example
        x=SX.sym("x")
        y=SX.sym("y")
        f = (1-x)**2+100*(y-x**2)**2
        nlp={'x':vertcat(x,y), 'f':f,'g':x+y}

        mycallback = MyCallback('mycallback', 2, 1, 0)
        self.mycallback = mycallback
        opts = {}
        opts['iteration_callback'] = mycallback
        opts['ipopt.tol'] = 1e-8
        opts['ipopt.max_iter'] = 50
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

        # first evaluation (successful)
        sol = self.solver(lbx=-10, ubx=10, lbg=-10, ubg=10)

        return None

# second evaluation (fail)
opti = Optimization()
sol = opti.solver(lbx=-10, ubx=10, lbg=-10, ubg=10)
