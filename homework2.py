from PlottingSystem import OCP_Plot

import matplotlib.pyplot as plt

from DAE import *


from casadi import *
import numpy as np

from enum import Enum
from copy import deepcopy
from typing import Union


class OptSolver(Enum):
    nlp = 0
    qp = 1
    sqp = qp


class OCP:
    def __init__(self, Name = ""):
        self.Name = Name
        pass

    def AddDAE(self, DAE_info:DAEInterface):
        assert False, " Deprecated! Use 'SetDAE' instead! "

    def SetDAE(self, DAE_info:DAEInterface):
        self.DAE_info:DAEInterface = DAE_info

        #in case no final condition is necessary, mask a 0-vector
        #self.xf_mask = DM( SX.ones( self.DAE_info.GetX().numel() ) )
        #also, the 'xf' is free ( e.g. doesn't influence the optimization process ) hence, it's value  doesn't matter
        #self.xf = DM( SX.ones( self.DAE_info.GetX().numel() ) )

        self.lbq = vertcat()
        self.ubq = vertcat()

        # time is always a symbol that you can choose to optimize for, or not,
        # if you choose to fix time then you need to have into 'Confi..' struct, tf=tf_lb=tf_ub
        #self.tf = SX.sym('tf',1)
        self.tf = type(self.DAE_info.xdot).sym('tf',1)

        return self
        pass

    def AddLagrangeCostFunction(self, L:Union[ MX, SX ]):
        if hasattr(self, "Lagrangian"):
            self.Lagranian = self.Lagranian + L
        else:
            self.Lagranian = L
        return self
        pass

    def SetLagrangeCostFunction(self, L:Union[ MX, SX ]):
        self.Lagranian = L
        return self
        pass

    def AddMayerCostFunction(self, M:Union[ MX, SX ]):

        var = symvar( M )
        assert len(var) == 1, "'final time' should be only one symbol! " \
                              "( mayer cost function is a function of final time ) "
        M = substitute( M, *var, self.tf )

        if hasattr(self, "Mayer"):
            self.Mayer = self.Mayer + M
        else:
            self.Mayer = M
        return self
        pass

    def SetMayerCostFunctionTo_0(self):

        #self.Mayer = SX.zeros(1,1)
        self.Mayer = type(self.DAE_info.xdot).zeros(1,1)
        return self
        pass

    def SetMayerCostFunction(self, M:Union[ MX, SX]):

        var = symvar( M )
        assert len(var) == 1, "'final time' should be only one symbol! " \
                              "( mayer cost function is a function of final time ) "
        M = substitute( M, *var, self.tf )

        self.Mayer = M

        return self
        pass


    #by default is 0
    def SetStartTime(self, t0:float = 0):
        self.t0 = t0
        return self
        pass

    # can be numeric or symbolic
    # if case of a symbolic value, the optimal control process get to be optimized w.r.t. final time "tf" too!
    def SetEndTime(self, tf:Union[float, MX, SX]):

        assert False, "Deprecated! If you are using it, you should reconsider your implementation! " \
                      "SetEndTime ( 'setting the end time' ) should be set into " \
                      "'Confi..' struct using tf=tf_lb=tf_ub !!! "

        self.tf = tf
        return self
        pass

    def SetX_0(self, x0, mask):
        self.x0 = x0
        self.x0_mask = mask
        return self
        pass

    def SetX_0_fromPreviousPhaseComputation(self):

        #assert False, "Not implemented yet!"
        assert hasattr(self, 'result'), \
            "'SetX_0_fromPreviousPhaseComputation' can be called starting with the second phase only!"

        # 'result' structure: (Sn, p) -> required: 'S_0' and 'p'
        self.x0 = self.eval_intg_exp_single_shooting( self.result[0][:,0], self.result[1] )[:,-1]

        # the new 'x0' must be equal with previous phase result 'xf'
        # ( multi-phase mathing conditions ): x0 - previous(xf) == 0, hence mask must be formed by 1s.
        self.x0_mask = [ 1 ] * len( self.x0_mask )

        return self
        pass

    def GetX_f_fromPreviousPhaseComputation(self):

        assert hasattr(self, 'result'), \
            "'GetX_f_fromPreviousPhaseComputation' can be called starting with the second phase only!"

        # 'result' structure: (Sn, p) -> required: 'S_0' and 'p'
        return self.eval_intg_exp_single_shooting( self.result[0][:,0], self.result[1] )[:,-1]
        pass

    def GetStatesAsFrames(self):

        assert hasattr(self, 'result'), "'GetStatesAsFrames' you didn't compute it yet!"

        # 'result' structure: (Sn, p) -> required: 'S_0' and 'p'
        return self.eval_intg_exp_single_shooting( self.result[0][:,0], self.result[1] )

        pass

    def GetEndStateOfFirstShootingInterval(self):

        assert hasattr(self, 'result'), "'GetEndStateOfFirstShootingInterval' you didn't compute it yet!"

        # 'result' structure: (Sn, p) -> required: 'S_0' and 'p'
        return self.eval_intg_exp_single_shooting( self.result[0][:,0], self.result[1] )[:,0]

    def GetEndStateOfFirstShootingIntervarl(self):

        assert False, "This is wrong!"

        assert hasattr(self, 'result'), "'GetEndStateOfFirstShootingInterval' you didn't compute it yet!"

        # 'result' structure: (Sn, p) -> required: 'S_0' and 'p'
        return Config_InvertedPendulumOnACard_NMPC['Sn']


    def GenerateFramesForFirstShooting25FPS(self):

        nr_Frames = 1

        assert hasattr(self, 'result'), "'GenerateFramesForFirstShooting25PerSec' you didn't compute it yet!"

        I_ForFrames = self.DAE_info_FrameGenerator.GetI_plot_TimeNormalized(
            self.t0, self.tf , self.NumberShootingNodes,
            NrEvaluationPoints = int((self.result[1][-1,0] * nr_Frames).full()[0][0]) )

        return I_ForFrames(x0=self.result[0][:,0], p=self.result[1][:,0] )['xf']


    def GetResulted_Sn_and_P(self):

        assert hasattr(self, 'result'), "'GetResulted_Sn_and_P' you didn't compute it yet!"

        return self.result

        pass

    def SetX_f(self, xf, mask):
        self.xf = xf
        self.xf_mask = mask
        return self
        pass

    def SetLB_Xf(self, eq, mask):
        self.Xf_lb = eq
        self.Xf_lb_mask = mask
        return self
        pass

    def SetUB_Xf(self, eq, mask):
        self.Xf_ub = eq
        self.Xf_ub_mask = mask
        return self
        pass

    #lower and upper bounds for control
    def SetLBW(self, lbw):
        self.lbw = lbw
        return self
        pass
    def SetUBW(self, ubw):
        self.ubw = ubw
        return self
        pass

    #lowe and upper bounds for parameters
    # constant parameter, variable parameters( e.g. final time )
    def SetLBQ(self, lbq):
        self.lbq = lbq
        return self
        pass
    def SetUBQ(self, ubq):
        self.ubq = ubq
        return self
        pass

    def SetLBtf(self, lb_tf):
        assert False, "I should think about it more..."
        self.lb_tf = lb_tf
        return self
        pass
    def SetUBtf(self, ub_tf):
        assert False, "I should think about it more..."
        self.ub_tf = ub_tf
        return self
        pass

    #set lower and upper bound functions for states
    def SetLB_State(self, eq, mask):
        self.lb_state = eq
        self.lb_state_mask = mask
        return self
        pass
    def SetUB_State(self, eq, mask):
        self.ub_state = eq
        self.ub_state_mask = mask
        return self
        pass

    def SetNumberShootingNodes(self, Number:int):
        self.NumberShootingNodes = Number
        return self
        pass

    def SetSolver(self, Solver=OptSolver.nlp):
        self.Solver = Solver
        return self
        pass

    def SetMaxIterationNumber(self, max_iter):
        self.max_iter = max_iter
        return self
        pass

    def BuildCasADiExpFromIntegrator(self, NumberShootingNodes, intg, SizeOfx0, SizeOfp, mapaccum=False):
        p = MX.sym('p', SizeOfp, NumberShootingNodes)

        if mapaccum == False:
            F = intg.map(NumberShootingNodes,"openmp")
            Sn = MX.sym('Sn', SizeOfx0, NumberShootingNodes)
            intg_exp = F( x0 = Sn, p=p )
        else:
            F = intg.mapaccum(NumberShootingNodes)
            Sn = MX.sym('Sn', SizeOfx0 )
            intg_exp = F( x0=Sn, p = p )

        return Sn, p, intg_exp

    def Plotter(self, Config, Sn_val, p_val , DAE_info ):

        #Sn_val = reshape(Sn_val,1,40)
        #p_val = reshape(p_val,1,40)
        #Sn_val = Sn_val.full()[0,:]
        #p_val = p_val.full()[0,:]

        if not isinstance(self.tf, (float, int)): # do it differently!
            tf = p_val[-1].full()[0]
            t0 = 0 #  hack! maybe t0 is not 0
        else:
            tf = self.tf
            t0 = self.t0

        #Plotting = OCP_Plot(self.t0, self.tf, self.NumberShootingNodes, DAE_info,
        #                    self.BuildCasADiExpFromIntegrator, 25, Config['SnName'])
        Plotting = OCP_Plot(t0, tf, self.NumberShootingNodes, DAE_info,
                            self.BuildCasADiExpFromIntegrator, 25, Config['SnName'])

        Plotting.Plot(Config['Sn'], Config['p'], False, 'Initial Guess', False)

        #Plotting.Plot(InitialGuess[0:2,:], InitialGuess[40:42,:], True, 'Iteration: 1', False)
        #Plotting.Plot(reshape(InitialGuess[:40,:],2,20), reshape(InitialGuess[40:,:],2,20), False, 'Iteration: 1', False)
        #Plotting.Plot(Config['Sn'], Config['p'], False, 'Iteration: 1', True)

        #Plotting.Plot(Config['Sn'], Config['p'], False, 'Iteration: 1', True)

        Plotting.Plot(Sn_val,p_val,False,'Iteration1', False)
        Plotting.Show()

        #plt.plot(np.linspace(0,Config['NumberShootingNodes'],Config['NumberShootingNodes']), Sn_val[0,:].T, '-')
        #plt.plot(np.linspace(0,Config['NumberShootingNodes'],Config['NumberShootingNodes']), Sn_val[1,:].T, '-.')
        #plt.show()

        #return self
        pass

    def PostProcessingAndPloting(self, result, Config, DAE_info, ActivePlotter = True):

        print(result['x'])

        res_sn = result['x'][:Config['Sn'].numel()]
        res_p = result['x'][Config['Sn'].numel():]
        Sn_val = reshape(res_sn, Config['Sn'].size1(),Config['Sn'].size2())
        p_val = reshape(res_p, Config['p'].size1(),Config['p'].size2())
        print("Sn: ", Sn_val[:,:])
        print("intg_in_multiple_shooting( Sn, p ): ", self.eval_intg_exp( Sn_val[:,:], p_val[:,:]) )
        print("intg_in_single_shooting( Sn, p ): ", self.eval_intg_exp_single_shooting( Sn_val[:,0], p_val[:,:] ) )
        print("P: ",p_val)

        if ActivePlotter:
            self.Plotter( Config, Sn_val, p_val, DAE_info )

        self.result = ( Sn_val[:,:], p_val ) # result as (Sn,p) in vector form each
        #return ( Sn_val[:,:], p_val ) # result as (Sn,p) in vector form each

    def NLP(self, X, f, g, lbg, ubg, InitialGuess, Config, DAE_info, max_iter = 100, ActivePlotter = True ):

        # https://projects.coin-or.org/CoinBinary/export/837/CoinAll/trunk/Installer/files/doc/Short%20tutorial%20Ipopt.pdf
        # https://coin-or.github.io/Ipopt/
        # https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_acceptable_tol
        # https://coin-or.github.io/Ipopt/OPTIONS.html
        # https://coin-or.github.io/Ipopt/SPECIALS.html
        # https://math.stackexchange.com/questions/199718/comparison-of-nonlinear-system-solvers
        # https://wiki.mcs.anl.gov/leyffer/images/7/75/NLP-Solvers.pdf
        # https://www.ccom.ucsd.edu/~elwong/talks/t/elw-ismp15.pdf
        # https://github.com/RobotLocomotion/drake/issues/1627
        # https://web.stanford.edu/class/msande318/notes/notes02-software.pdf
        # https://github.com/casadi/casadi/blob/master/docs/api/examples/solvers/callback.py -> access nlpsol iterations via callback
        # https://groups.google.com/g/casadi-users/c/oO-y_f7X-s4/m/35VLJLtMAgAJ -> access nlpsol iterations via callback
        # test19_callback_NLP.py

        nlp = {'x': X,
               'f': f,
               'g': g
               }

        SolverNLP = nlpsol('S', 'ipopt', nlp,
                           {
                               'error_on_fail':False,
                               'eval_errors_fatal':True,
                               'verbose':True,
                               #'expand':True,
                               #'ipopt.acceptable_tol': 1e-12,
                               #'ipopt.mumps_pivtol':1e-12,
                               #'ipopt.tol':1e-13,
                               #'ipopt.acceptable_constr_viol_tol':1e-12
                               #'ipopt.acceptable_constr_viol_tol':1e-5,
                               #'monitor':'nlp_g',
                               'ipopt.hessian_approximation':'limited-memory', # https://coin-or.github.io/Ipopt/OPTIONS.html
                               'ipopt.max_iter' : max_iter,
                           } )

        result = SolverNLP(
            x0= InitialGuess,
            #lbx=0,
            #ubx=0,
            lbg=lbg,
            ubg=ubg
        )


        ###lam_x = result['lam_x']
        ###lam_g = result['lam_g']

        # Set options
        ###opts = {}
        #opts['max_iter'] = 60
        ###opts['verbose'] = False
        ###opts['hessian_approximation'] = 'gauss-newton' #'exact'

        #Specify QP solver
        ###opts['qpsol']  = 'qpoases' #'nlpsol' #'qrqp' # 'nlpsol'
        #opts['qpsol_options.nlpsol'] = 'ipopt'
        ###opts['qpsol_options.error_on_fail'] = False
        #opts['qpsol_options.nlpsol_options.ipopt.print_level'] = 0
        #opts['qpsol_options.nlpsol_options.print_time'] = 0
        ###opts['qpsol_options'] = {"printLevel":"none"}

        ###SolverSQP = nlpsol('SolverSQP', 'sqpmethod', nlp, opts)
        #SolverSQP = nlpsol('SolverSQP', 'scpgen', nlp, opts)  # -> in development .. structured SQP

        ###SolverSQP = nlpsol('SolverSQP', 'blocksqp', nlp, opts) # -> requires libcasadi_linsol_ma27x
        ###   -> https://github-wiki-see.page/m/casadi/casadi/wiki/Obtaining-HSL
        ###   -> https://github.com/casadi/casadi/issues/1932
        ###   -> requires 'libhsl.dll'
        ###   -> how to: https://groups.google.com/g/casadi-users/c/JtkBc5Gy2r8/m/sxiTjkPwAAAJ
        ###   -> https://github.com/casadi/casadi/wiki/FAQ:-how-to-get-third-party-solvers-to-work%3F


        ###result = SolverSQP( x0= InitialGuess,
        ###                    #lam_x0= lam_x, lam_g0= lam_g,
        ###                    #lbx=0, #ubx=0,
        ###                    lbg=lbg, ubg=ubg
        ###                    )


        #return self.PostProcessingAndPloting( result, Config, DAE_info, ActivePlotter )
        self.PostProcessingAndPloting( result, Config, DAE_info, ActivePlotter )
        pass

    def SQP(self, X, f, g, lbg, ubg, InitialGuess, Config, DAE_info, max_iter = 10, ActivePlotter=True):

        nlp = {'x': X,
               'f': f,
               'g': g
               }

        #lam_x = result['lam_x']
        #lam_g = result['lam_g']

        # Set options
        opts = {}
        opts['max_iter'] = max_iter
        opts['verbose'] = False
        opts['hessian_approximation'] = 'gauss-newton' #'exact'

        #Specify QP solver
        opts['qpsol']  = 'qpoases' #'nlpsol' #'qrqp' # 'nlpsol'
        #opts['qpsol_options.nlpsol'] = 'ipopt'
        opts['qpsol_options.error_on_fail'] = False
        #opts['qpsol_options.nlpsol_options.ipopt.print_level'] = 0
        #opts['qpsol_options.nlpsol_options.print_time'] = 0
        opts['qpsol_options'] = {"printLevel":"none"}

        SolverSQP = nlpsol('SolverSQP', 'sqpmethod', nlp, opts)
        #SolverSQP = nlpsol('SolverSQP', 'scpgen', nlp, opts)  # -> in development .. structured SQP

        #SolverSQP = nlpsol('SolverSQP', 'blocksqp', nlp, opts) # -> requires libcasadi_linsol_ma27x
        #   -> https://github-wiki-see.page/m/casadi/casadi/wiki/Obtaining-HSL
        #   -> https://github.com/casadi/casadi/issues/1932
        #   -> require 'libhsl.dll'
        #   -> how to: https://groups.google.com/g/casadi-users/c/JtkBc5Gy2r8/m/sxiTjkPwAAAJ
        #   -> https://github.com/casadi/casadi/wiki/FAQ:-how-to-get-third-party-solvers-to-work%3F


        result = SolverSQP( x0= InitialGuess,
                            #lam_x0= lam_x, lam_g0= lam_g,
                            #lbx=0, #ubx=0,
                            lbg=lbg, ubg=ubg
                            )

        #return self.PostProcessingAndPloting(result, Config, DAE_info, ActivePlotter)
        self.PostProcessingAndPloting(result, Config, DAE_info, ActivePlotter)
        pass

    def SQP_manual(self, X, f, g, lbg, ubg, InitialGuess, ConfigInput, DAE_info, ActivePlotter):

        assert False, "Deprecated! Please consider using SQP()"

        Config= {
            'x':X,
            'f':f,
            'g':g,
            'lbg': lbg,
            'ubg': ubg,
            'InitialGuess': InitialGuess
        }

        #A = DM.rand( Config['x'].numel(), Config['x'].numel() )
        A = DM.eye(Config['x'].numel())
        psd = 1/2 * (A + A.T)

        H = psd
        Grad = jacobian( Config['f'], Config['x'] )

        #design path... :(
        if Grad.size1()==0:
            Grad = MX.zeros( Grad.size2() ).T

        DeltaX = MX.sym('DeltaX', Config['x'].numel(), 1)

        #inf >= lbg >= 0
        #lbg_ieq_exp = Config['g'] - Config['lbg']
        #inf >= ubg >= 0
        #ubg_ieq_exp = Config['ubg'] - Config['g']
        #ieq_exp = vertcat(lbg_ieq_exp, ubg_ieq_exp)
        ieq_exp = Config['g']

        # no need, already in the "inf>=g>=0" form!!!
        #lbg = vertcat( DM.zeros( lbg_ieq_exp.size1() ), DM.zeros( ubg_ieq_exp.size1() ) )
        #ubg = vertcat( DM.inf( lbg_ieq_exp.size1() ), DM.inf( ubg_ieq_exp.size1() ) )


        #first order approximation
        #vars = symvar(ieq_exp)
        #g_FOA.call()
        #vars = symvar( Config['x'] )
        #cost_exp_eval = Function( 'cost_exp_eval', vars, [cost_exp] )
        #cost = cost_exp_eval.call
        #linearized_const = linearize( ieq_exp, Config['x'], Config['InitialGuess'] )
        #linearized_const = linearize( ieq_exp, Config['x'],  )

        ###if not isinstance(self.tf, float):
        ###    # do it differently
        ###    tf = p_val[-1].full()[0]
        ###    t0 = 0 #  hack! maybe t0 is not 0
        ###else:
        ###    tf = self.tf
        ###    t0 = self.t0

        ###Plotting = OCP_Plot(t0, tf, self.NumberShootingNodes, DAE_info,
        ###                    self.BuildCasADiExpFromIntegrator, 25, Config['SnName'])

        ###Plotting.Plot(Config['Sn'], Config['p'], False, 'Initial Guess', True)

        for indx in range(0,30):

            cost_exp = 1/2 * DeltaX.T @ H @ DeltaX + dot(Grad.T, DeltaX)
            cost = substitute( cost_exp, Config['x'], DM(Config['InitialGuess']) )

            linearized_constraints = ieq_exp + jacobian( ieq_exp, Config['x'] ) @ DeltaX
            constraints = substitute(linearized_constraints, Config['x'], DM(Config['InitialGuess']) )

            qp = { 'x': DeltaX, 'f': cost, 'g': constraints }
            S = qpsol('S', 'qpoases', qp, {'error_on_fail':False})
            print(S)
            if 'lam_x' in Config:
                result = S(
                    x0=Config['InitialGuess'],
                    lbg=lbg, ubg=ubg, lam_x0=Config['lam_x'], lam_g0=Config['lam_g'])
            else:
                result = S(x0=Config['InitialGuess'], lbg=lbg, ubg=ubg)

            Config['InitialGuess'] = Config['InitialGuess'] + 0.3*result['x']

            Config['lam_x'] = result['lam_x']
            Config['lam_g'] = result['lam_g']

            #Lagrangian  = Config['f'] - Config['lam_g'] * ieq_exp
            #H = jacobian( jacobian(Lagrangian, Config['x']), Config['x'] )

            #print(sol)
            #print(Config['InitialGuess'])

            res_sn = result['x'][:ConfigInput['Sn'].numel()]
            res_p = result['x'][ConfigInput['Sn'].numel():]
            Sn_val = reshape(res_sn, ConfigInput['Sn'].size1(),ConfigInput['Sn'].size2())
            p_val = reshape(res_p, ConfigInput['p'].size1(),ConfigInput['p'].size2())
            print('Sn:', Sn_val[:,:])
            print("intg(Sn,p):", self.eval_intg_exp(Sn_val[:,:], p_val[:,:]))
            print("P:",p_val)

            #plt.plot(np.linspace(0,ConfigInput['NumberShootingNodes'],ConfigInput['NumberShootingNodes']),
            #         Sn_val[0,:].T, '-')
            #plt.plot(np.linspace(0,ConfigInput['NumberShootingNodes'],ConfigInput['NumberShootingNodes']),
            #         Sn_val[1,:].T, '-.')
            #plt.show()

            ###Plotting.Plot(Sn_val,p_val,False,'Iteration: '+ str(indx), False)

        ###Plotting.Show()
        pass

    def Build(self, Config, TrueSolutionS_i_W_i_Q_i = None, ActivePlotter = True , resetInitialGuess = False):

        assert hasattr(self, "tf"), "no tf set!"

        Config = deepcopy( Config )
        #DAE_info = deepcopy( self.DAE_info )       #doesn't work with homework 4
        self.DAE_info.ResetCurrentQuadrature()      # used for homework 4
        DAE_info = self.DAE_info                    # used for homework 4

        # first, transform Mayer term into Lagrange term and add it to current Lagrange term.!!!
        if hasattr(self, "Mayer"):
            # it will crash when "tf" is not a symbol ( no variable end time ) and we have Mayer term: M(tf=const)
            # an option is to add the Mayer fix term directly to F1_exp by ( Already done! )
            # adding the 'qf'( eval_quadrature_exp ) to the result of Mayer evaluation function.
            if hasattr(self, "Lagranian") == False: self.Lagranian = 0
            self.Lagranian = jacobian(self.Mayer, self.tf) + self.Lagranian

        if hasattr(self, "Lagranian"):
            DAE_info.AddQuadrature( self.Lagranian )

        #NO needed anymore, as this call can be considered a special case of the next call!
        #intg = DAE_info.GetI( self.t0, self.tf / self.NumberShootingNodes )
        intg = DAE_info.GetI_TimeNormalized( self.t0, self.tf , self.NumberShootingNodes )
        #from this point on, 'tf' must be considered '1' after normalization!!!

        self.DAE_info_FrameGenerator = DAE_info

        # if it is a symbol than we must optimize for it, hence, "tf" and it's value must be added
        # as part of "q" in "Config" to be used as initial guess for "variable" parameters.
        if isinstance( self.tf, ( SX, MX ) ):
            assert 'tf' in Config, "Must have initial value 'tf' for final time or to optimize for/ fixed final time!"
            assert 'tf_lb' in Config, "Config struct is missing 'tf_lb'! ( for fixed 'tf', 'tf_lb=tf' ) "
            assert 'tf_ub' in Config, "Config struct is missing 'tf_ub'! ( for fixed 'tf', 'tf_ub=tf' ) "
            Config['q'] = Config['q'] + [ Config['tf'] ]
            Config['qName'] = Config['qName'] + ['tf']
            Config['lbq'] = Config['lbq'] + [Config['tf_lb']]
            Config['ubq'] = Config['ubq'] + [Config['tf_ub']]
            self.lbq = vertcat(self.lbq, Config['tf_lb'])
            self.ubq = vertcat(self.ubq, Config['tf_ub'])

        # extract initial values for S_i, w_i, q_i from previous phase
        # ( the result of previous phase is maintain as an internal state )
        if (hasattr(self, 'result') and (not resetInitialGuess)):
            assert False, "Not implemented yet!"
            Sn_InitialGuess = self.result[0]
            p_InitialGuess = self.result[1]
            pass
        #generate initial values for each S_i, w_i, q_i
        elif TrueSolutionS_i_W_i_Q_i == None:
            # in this situation, w_i and q_i are the same for the entire interval
            # in this case, we only compute "S_i"

            Sn_aux, p_aux, intg_exp_aux = self.BuildCasADiExpFromIntegrator(
                NumberShootingNodes = self.NumberShootingNodes,
                intg = intg,
                SizeOfx0 = DAE_info.GetX().numel(),
                SizeOfp = DAE_info.GetP().numel(),
                mapaccum = True )

            eval_intg_exp_aux=Function('eval_intg_exp_aux',
                                       [Sn_aux, p_aux], [intg_exp_aux['xf']], ['x0', 'p'], ['out'])
            # maintain this function in case I need to evaluate the final solution ( e.g. generating
            # a nice car-inverted pendulum drawing )
            self.eval_intg_exp_single_shooting = eval_intg_exp_aux

            Sn_InitialGuess = eval_intg_exp_aux( Config['Sn'], Config['w'] + Config['q'] )
            Sn_InitialGuess = horzcat(Config['Sn'], Sn_InitialGuess[:,:-1])
            #w_InitialGuess = repmat( DM( Config['w'] ), 1, self.NumberShootingNodes )
            #q_InitialGuess = repmat( DM( Config['q'] ), 1, self.NumberShootingNodes )
            p_InitialGuess = repmat( vertcat( DM( Config['w'] ), DM( Config['q'] ) ), 1, self.NumberShootingNodes )
        else:
            #assert False, "Not implemented yet! (required by problems with extra inequalities constraint" \
            #              "where is not 'easy' to use simulation as a way to generate initial guess, e.g.: car!)"
            Sn_InitialGuess = TrueSolutionS_i_W_i_Q_i[0] #TrueSolutionS_i_W_i_Q_i.SnInitialGuess #maybe se have to delete the first node!!! :)
            #w_InitialGuess = TrueSolutionS_i_W_i_Q_i.w_InitialGuess
            #q_InitialGuess = TrueSolutionS_i_W_i_Q_i.q_InitialGuess
            p_InitialGuess = TrueSolutionS_i_W_i_Q_i[1] #TrueSolutionS_i_W_i_Q_i.p_InitialGuess
            pass

        Config['Sn'] = Sn_InitialGuess
        Config['p'] = p_InitialGuess

        Sn, p, intg_exp = self.BuildCasADiExpFromIntegrator( NumberShootingNodes = self.NumberShootingNodes,
                                                             intg = intg,
                                                             SizeOfx0 = DAE_info.GetX().numel(),
                                                             SizeOfp = DAE_info.GetP().numel(),
                                                             mapaccum = False )

        eval_intg_exp = Function('eval_intg_exp', [Sn, p], [intg_exp['xf']], ['x0', 'p'], ['out'])
        self.eval_intg_exp = eval_intg_exp

        eval_quadrature_exp = Function('quadrature_exp',[Sn, p], [intg_exp['qf']], ['x0', 'p'], ['out'])

        #cost function
        F1_exp = sum2(eval_quadrature_exp( Sn, p ))

        #matching equality constraints
        F2_exp = eval_intg_exp(Sn, p)[:, :-1] - Sn[:, 1:]
        #F2_exp = vertcat( F2_exp[0:2,:], F2_exp[5,:] )
        #mask = MX([1,1,0,0,0,1,0])
        #F2_exp = F2_exp * repmat( mask, 1, F2_exp.size2() ) # we don't need continuity for all states.!
        F2_exp = reshape( F2_exp, F2_exp.numel(), 1)
        #F2_exp = reshape( F2_exp, 3*F2_exp.size2(), 1)

        lbg = DM.zeros( F2_exp.size1() )
        ubg = DM.zeros( F2_exp.size1() )
        #F2_exp = vertcat()
        #lbg = vertcat()
        #ubg = vertcat()

        #create equality constraint for S_0: Sn[0] - x0 == 0 given the mask: x0_mask
        if hasattr(self, 'x0') and hasattr(self, 'x0_mask'):
            x0_eq_constraint = Sn[:, 0] * DM( self.x0_mask ) - self.x0 * DM( self.x0_mask )
            #add x_eq_constraint into F2_exp
            F2_exp = vertcat( F2_exp, x0_eq_constraint )

            lbg = vertcat( lbg, DM.zeros( x0_eq_constraint.size1() ) )
            ubg = vertcat( ubg, DM.zeros( x0_eq_constraint.size1() ) )

        assert not ( hasattr(self, 'xf') and hasattr(self, 'Xf_lb') ), \
            "Can't have fixed final: 'xf' and lower bound final: 'Xf_lb' in the same time!"

        assert not ( hasattr(self, 'xf') and hasattr(self, 'Xf_ub') ), \
            "Can't have fixed final: 'xf' and upper bound final: 'Xf_ub' in the same time!"

        #create equality constraint for xf: intg( Sn[-1] ) - xf == 0 given the mask: xf_mask
        if hasattr(self, 'xf') and hasattr(self, 'xf_mask'):
            xf_eq_constraint = eval_intg_exp(Sn, p)[:, -1] * DM( self.xf_mask ) - self.xf * DM( self.xf_mask )
            #add xf_eq_constraint into F2_exp
            F2_exp = vertcat( F2_exp, xf_eq_constraint )

            lbg = vertcat( lbg, DM.zeros( xf_eq_constraint.size1() ) )
            ubg = vertcat( ubg, DM.zeros( xf_eq_constraint.size1() ) )


        assert (hasattr(self, 'Xf_lb') and hasattr(self, 'Xf_lb_mask')) or \
               (not hasattr(self, 'Xf_lb') and not hasattr(self, 'Xf_lb_mask')), \
            "You must have 'Xf_lb' as well as 'Xf_lb_mask' defined or non of them!"

        #lower bounds for 'xf': inf >= lb_state_xf_exp >= 0 given the mask: 'Xf_lb_mask'
        if hasattr(self, 'Xf_lb') and hasattr(self, 'Xf_lb_mask'):
            _xf = horzcat( eval_intg_exp(Sn, p)[:, -1] )  # [ xf ]
            dummy_variable = MX.sym('dummy_variable', self.DAE_info.GetX().numel() ) #Sn_and_xf[:,0].numel() ) # only used for generating 'lb_state'
            lb_state_xf = self.Xf_lb( dummy_variable ).map( _xf.size2() ) # vector form (only one element in this case as _xf.size2() is '1')

            lb_state_xf_exp = _xf * repmat( DM( self.Xf_lb_mask ), 1, _xf.size2() ) - \
                              lb_state_xf( _xf ) * repmat( DM( self.Xf_lb_mask ), 1, _xf.size2() )

            lb_state_xf_exp = reshape( lb_state_xf_exp, lb_state_xf_exp.numel(), 1 )

            F2_exp = vertcat( F2_exp, lb_state_xf_exp )

            lbg = vertcat( lbg, DM.zeros( lb_state_xf_exp.size1() ) )
            ubg = vertcat( ubg, DM.inf( lb_state_xf_exp.size1() ) )


        assert (hasattr(self, 'Xf_ub') and hasattr(self, 'Xf_ub_mask')) or \
               (not hasattr(self, 'Xf_ub') and not hasattr(self, 'Xf_ub_mask')), \
            "You must have 'Xf_ub' as well as 'Xf_ub_mask' defined or non of them!"

        #upper bounds for 'xf': inf >= ub_state_xf_exp >= 0 given the mask: 'Xf_ub_mask'
        if hasattr(self, "Xf_ub") and hasattr(self, 'Xf_ub_mask'):
            _xf = horzcat( eval_intg_exp(Sn, p)[:, -1] )  # [ xf ]
            dummy_variable = MX.sym('dummy_variable', self.DAE_info.GetX().numel() ) #Sn_and_xf[:,0].numel() ) # only used for generating 'ub_state'
            ub_state_xf = self.Xf_ub( dummy_variable ).map( _xf.size2() ) # vector form (only one element in this case as _xf.size2() is '1')

            ub_state_xf_exp = ub_state_xf( _xf ) * repmat( DM( self.Xf_ub_mask ), 1, _xf.size2() ) - \
                           _xf * repmat( DM(  self.Xf_ub_mask  ), 1, _xf.size2() )

            ub_state_xf_exp = reshape( ub_state_xf_exp, ub_state_xf_exp.numel(), 1 )

            F2_exp = vertcat( F2_exp, ub_state_xf_exp )

            lbg = vertcat( lbg, DM.zeros( ub_state_xf_exp.size1() ) )
            ubg = vertcat( ubg, DM.inf( ub_state_xf_exp.size1() ) )


        #add q_i - q_{i-1} if necessary ( it is... in case "tf" variates... currently, variation of other parameter
        # and optimization for them is not supported! ( changes to be made! )

        #equality constrain for all intervals for "time": q_i [ time ] - q_{i-1}[ time ] == 0
        if isinstance( self.tf, ( SX, MX ) ):
            #assert False, "I didn't check this case!"
            #position of time as part of "P"
            #DAE_info.GetP().numel()
            mask = vertcat( DM.zeros( DAE_info.GetP().numel() - 1 ) , 1 )
            vec_of_time_param = p * repmat( mask, 1, p.size2() )
            equality_time_exp = vec_of_time_param[-1,:-1] - vec_of_time_param[-1,1:]

            equality_time_exp = reshape(equality_time_exp, equality_time_exp.numel(), 1)

            F2_exp = vertcat( F2_exp, equality_time_exp )

            lbg = vertcat( lbg, DM.zeros( equality_time_exp.numel() ) )
            ubg = vertcat( ubg, DM.zeros( equality_time_exp.numel() ) )

        #add multipoint boundary constraint
        if hasattr(self, 'multipoint'):
            assert False, "Multipoint boundary constraint not yet implemented!"
            # https://mashayekhi.iut.ac.ir/sites/mashayekhi.iut.ac.ir/files//files_course/lesson_16.pdf
            # https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node110.html
            # https://abaqus-docs.mit.edu/2017/English/SIMACAECSTRefMap/simacst-c-mpc.htm#simacst-c-mpc-t-UseWithTransformedCoordinateSystems-sma-topic2
            # https://www.youtube.com/watch?v=8j4Z0IrS96Q
            # https://www.astos.de/products/sos/details
            # Personally, I would try to implement it using splines based on multiple shooting nodes.

        # transformed into: inf >= lb_state_exp >= 0
        if hasattr(self, "lb_state"):
            Sn_and_xf = horzcat( Sn, eval_intg_exp(Sn, p)[:, -1] )  # [ Sn xf ]
            dummy_variable = MX.sym('dummy_variable', self.DAE_info.GetX().numel() ) #Sn_and_xf[:,0].numel() ) # only used for generating 'lb_state'
            lb_state = self.lb_state( dummy_variable ).map( Sn_and_xf.size2() ) # vector form

            lb_state_exp = Sn_and_xf * repmat( DM(  self.lb_state_mask  ), 1, Sn_and_xf.size2() ) - \
                           lb_state( Sn_and_xf ) * repmat( DM( self.lb_state_mask ), 1, Sn_and_xf.size2() )

            lb_state_exp = reshape(lb_state_exp, lb_state_exp.numel(), 1)

            F2_exp = vertcat(F2_exp, lb_state_exp)

            lbg = vertcat( lbg, DM.zeros( lb_state_exp.size1() ) )
            ubg = vertcat( ubg, DM.inf( lb_state_exp.size1() ) )


        # transformed into: inf >= ub_state_exp >= 0
        if hasattr(self, "ub_state"):
            Sn_and_xf = horzcat( Sn, eval_intg_exp(Sn, p)[:, -1] )  # [ Sn xf ]
            dummy_variable = MX.sym('dummy_variable', self.DAE_info.GetX().numel() ) #Sn_and_xf[:,0].numel() ) # only used for generating 'ub_state'
            ub_state = self.ub_state( dummy_variable ).map( Sn_and_xf.size2() ) # vector form

            ub_state_exp = ub_state( Sn_and_xf ) * repmat( DM( self.ub_state_mask ), 1, Sn_and_xf.size2() ) - \
                           Sn_and_xf * repmat( DM(  self.ub_state_mask  ), 1, Sn_and_xf.size2() )

            ub_state_exp = reshape(ub_state_exp, ub_state_exp.numel(), 1)

            F2_exp = vertcat( F2_exp, ub_state_exp)

            lbg = vertcat( lbg, DM.zeros( ub_state_exp.size1() ) )
            ubg = vertcat( ubg, DM.inf( ub_state_exp.size1() ) )


        ###
        ## control and constant parameter upper and lower bounds
        ###

        assert (hasattr(self, 'lbw') and hasattr(self, 'lbq')) or \
               (not hasattr(self, 'lbw') and not hasattr(self, 'lbq')), "You must have 'lbw' as well as 'lbq' defined" \
                                                                        "or non of them!"
        #lower bounds for "w" and "q": inf >= vec_lb_p_exp >= 0
        if hasattr(self, 'lbw') and hasattr(self, 'lbq'):
            lb_p = vertcat( self.lbw, self.lbq )
            vec_lb_p = repmat( lb_p, 1, p.size2() )
            vec_lb_p_exp = p - vec_lb_p
            vec_lb_p_exp = reshape( vec_lb_p_exp, vec_lb_p_exp.numel(), 1 )
            F2_exp = vertcat(F2_exp, vec_lb_p_exp)

            lbg = vertcat( lbg, DM.zeros( vec_lb_p_exp.size1() ) )
            ubg = vertcat( ubg, DM.inf( vec_lb_p_exp.size1() ) )


        assert (hasattr(self, 'ubw') and hasattr(self, 'ubq')) or \
               (not hasattr(self, 'ubw') and not hasattr(self, 'ubq')), \
            "You must have 'ubw' as well as 'ubq' defined or non of them!"

        #upper bounds for "w" and "q": inf >= vec_ub_p_exp >= 0
        if hasattr(self, 'ubw') and hasattr(self, 'ubq'):
            ub_p = vertcat( self.ubw, self.ubq )
            vec_ub_p = repmat( ub_p, 1, p.size2() )
            vec_ub_p_exp = vec_ub_p - p
            vec_ub_p_exp = reshape( vec_ub_p_exp, vec_ub_p_exp.numel(), 1 )
            F2_exp = vertcat( F2_exp, vec_ub_p_exp)

            lbg = vertcat( lbg, DM.zeros( vec_ub_p_exp.size1() ) )
            ubg = vertcat( ubg, DM.inf( vec_ub_p_exp.size1() ) )


        X = vertcat( reshape( Sn, Sn.numel(), 1), reshape( p, p.numel(), 1) )
        InitialGuess = vertcat( reshape( Config['Sn'], Config['Sn'].numel(), 1),
                                reshape( Config['p'], Config['p'].numel(), 1 ) )

        # 'lbx and ubx' not used!
        #X, f, g, lbg, ubg, InitialGuess
        if self.Solver == OptSolver.nlp:
            ##result =
            self.NLP( X = X, f = F1_exp, g = F2_exp, lbg = lbg, ubg = ubg, InitialGuess = InitialGuess,
                      Config = Config, DAE_info = DAE_info, max_iter = self.max_iter, ActivePlotter = ActivePlotter ) # max_iter = 70

        if self.Solver == OptSolver.qp:
            ##result =
            self.SQP( X = X, f = F1_exp, g = F2_exp, lbg = lbg, ubg = ubg, InitialGuess = InitialGuess,
                      Config = Config, DAE_info = DAE_info, max_iter = self.max_iter, ActivePlotter = ActivePlotter ) # max_iter = 20
        ##    result  = self.SQP_manual(X = X, f = F1_exp, g = F2_exp, lbg = lbg, ubg = ubg, InitialGuess = InitialGuess,
        ##                              ConfigInput = Config, DAE_info = DAE_info, max_iter = self.max_iter, ActivePlotter = ActivePlotter )

        #self.result = result
        return self
        pass

#Car_ODE = ClassDefineOneShootingNodForCar("Car")
#I = Car_ODE.GetI( 0, 0.1 )
#print(GroundTruthCar['Sn'])
#GroundTruthCar['Sn'] = \
#    I( x0 = [-29.2477, 1.33793e-020, 5.04412, -5.76646e-022, -8.92924e-021, -6.19568e-021, 1.59348e-021], p = GroundTruthCar['w'] + GroundTruthCar['q'] )['xf']
#print(GroundTruthCar['Sn'])
#GroundTruthCar['Sn'] = \
#    I( x0 = GroundTruthCar['Sn'], p = GroundTruthCar['w'] + GroundTruthCar['q'] )['xf']
#print(GroundTruthCar['Sn'])


#VanDerPol_ODE = ClassDefineOneShootingNodForVanDerPol("VanDerPol")
##sumsqr( VanDerPol_ODE.GetW() ) + sumsqr( VanDerPol_ODE.GetX() )
#ocp = OCP( "Optimize VanDerPol" ). \
#    AddDAE( VanDerPol_ODE ). \
#    AddLagrangeCostFunction( L = sumsqr( VanDerPol_ODE.GetW() ) + sumsqr( VanDerPol_ODE.GetX() ) ). \
#    SetStartTime( t0 = Config_VanDerPol['t0'] ). \
#    SetEndTime( tf = Config_VanDerPol['tf'] ). \
#    SetX_0( x0 = Config_VanDerPol['Sn'], mask=Config_VanDerPol['S0_mask'] ). \
#    SetX_f( xf = Config_VanDerPol['Xf'], mask=Config_VanDerPol['Xf_mask'] ). \
#    SetLBW(lbw=Config_VanDerPol['lbw'] ). \
#    SetUBW(ubw=Config_VanDerPol['ubw'] ). \
#    SetLBQ(lbq=Config_VanDerPol['lbq']). \
#    SetUBQ(ubq=Config_VanDerPol['ubq']). \
#    SetNumberShootingNodes( Number = Config_VanDerPol['NumberShootingNodes'] ). \
#    SetSolver( Solver = OptSolver.nlp )
##    Build()
#ocp.Build( Config_VanDerPol )

## sumsqr( LotkaVolterra_ODE.GetX() ) + sumsqr( LotkaVolterra_ODE.GetW() )
#LotkaVolterra_ODE = ClassDefineLotka_Volterra("LotkaVolterra")
#ocp = OCP( "Optimize LotkaVolterra" ). \
#    AddDAE( LotkaVolterra_ODE ). \
#    AddLagrangeCostFunction( L = sumsqr( LotkaVolterra_ODE.GetX() ) + sumsqr( LotkaVolterra_ODE.GetW() ) ). \
#    SetStartTime( t0 = Config_LotkaVolterra['t0'] ). \
#    SetEndTime( tf = Config_LotkaVolterra['tf'] ). \
#    SetX_0( x0 = Config_LotkaVolterra['Sn'], mask=Config_LotkaVolterra['S0_mask'] ). \
#    SetX_f( xf = Config_LotkaVolterra['Xf'], mask=Config_LotkaVolterra['Xf_mask'] ). \
#    SetLBW(lbw=Config_LotkaVolterra['lbw'] ). \
#    SetUBW(ubw=Config_LotkaVolterra['ubw'] ). \
#    SetLBQ(lbq=Config_LotkaVolterra['lbq']). \
#    SetUBQ(ubq=Config_LotkaVolterra['ubq']). \
#    SetNumberShootingNodes( Number = Config_LotkaVolterra['NumberShootingNodes'] ). \
#    SetSolver( Solver = OptSolver.nlp )
##    Build()
#ocp.Build( Config_LotkaVolterra )

#tf = SX.sym('tf',1)
#Mayer_exp = tf
#Brachistochrone_ODE = ClassDefineBrachistochrone("Brachistochrone")
#ocp = OCP( "Optimize Brachistochrone" ).\
#    AddDAE(Brachistochrone_ODE).\
#    AddLagrangeCostFunction( L = 0 ).\
#    AddMayerCostFunction( M = Mayer_exp ).\
#    SetStartTime( t0 = Config_Brachistochrone['t0'] ).\
#    SetEndTime( tf = tf ) .\
#    SetX_0( x0 = Config_Brachistochrone['Sn'], mask = Config_Brachistochrone['S0_mask'] ).\
#    SetX_f( xf = Config_Brachistochrone['Xf'], mask = Config_Brachistochrone['Xf_mask'] ).\
#    SetLBW( lbw = Config_Brachistochrone['lbw'] ).\
#    SetUBW( ubw = Config_Brachistochrone['ubw'] ). \
#    SetLB_State( eq=Config_Brachistochrone['lb_state'] , mask=Config_Brachistochrone['lb_state_mask'] ). \
#    SetUB_State( eq=Config_Brachistochrone['ub_state'] , mask=Config_Brachistochrone['ub_state_mask'] ). \
#    SetNumberShootingNodes( Number = Config_Brachistochrone['NumberShootingNodes'] ).\
#    SetSolver( Solver = OptSolver.nlp )
#ocp.Build(Config_Brachistochrone)


#tf = SX.sym('tf',1)
#Mayer_exp = tf
#Rocket_Car_ODE = ClassDefineRocket_Car("Rocket_Car")
#ocp = OCP( "Optimize Rocket_Car" ).\
#    AddDAE(Rocket_Car_ODE).\
#    AddMayerCostFunction( M = Mayer_exp ). \
#    SetStartTime( t0 = Config_Rocket_Car['t0'] ).\
#    SetEndTime( tf = tf ).\
#    SetX_0( x0 = Config_Rocket_Car['Sn'], mask = Config_Rocket_Car['S0_mask'] ).\
#    SetX_f( xf = Config_Rocket_Car['Xf'], mask = Config_Rocket_Car['Xf_mask'] ).\
#    SetLBW( lbw = Config_Rocket_Car['lbw'] ).\
#    SetUBW( ubw = Config_Rocket_Car['ubw'] ).\
#    SetNumberShootingNodes( Number = Config_Rocket_Car['NumberShootingNodes'] ).\
#    SetSolver( Solver = OptSolver.nlp )
#ocp.Build(Config_Rocket_Car)

#tf = SX.sym('tf',1)
#Mayer_exp = tf
#Car_ODE = ClassDefineOneShootingNodForCar("Car")
##sumsqr( Car_ODE.GetW() ) + sumsqr( Car_ODE.GetX() )
#ocp = OCP( "Optimize Simulated Car" ). \
#    AddDAE( Car_ODE ). \
#    AddLagrangeCostFunction( L = 0 ). \
#    AddMayerCostFunction( M = Mayer_exp ). \
#    SetStartTime( t0 = GroundTruthCar['t0'] ). \
#    SetEndTime( tf = tf ). \
#    SetX_0( x0 = GroundTruthCar['Sn'], mask=GroundTruthCar['S0_mask'] ). \
#    SetX_f( xf = GroundTruthCar['Xf'], mask=GroundTruthCar['Xf_mask'] ). \
#    SetLBW(lbw=GroundTruthCar['lbw'] ). \
#    SetUBW(ubw=GroundTruthCar['ubw'] ). \
#    SetLBQ(lbq=GroundTruthCar['lbq'] ). \
#    SetUBQ(ubq=GroundTruthCar['ubq'] ). \
#    SetLB_State( eq=GroundTruthCar['lb_state'] , mask=GroundTruthCar['lb_state_mask'] ). \
#    SetUB_State( eq=GroundTruthCar['ub_state'] , mask=GroundTruthCar['ub_state_mask'] ). \
#    SetNumberShootingNodes( Number = GroundTruthCar['NumberShootingNodes'] ). \
#    SetSolver( Solver = OptSolver.nlp )
##    Build()
#ocp.Build( GroundTruthCar )


##### >>>>>>>

#tf = SX.sym('tf',1)
#Mayer_exp = tf
#Car_ODE = ClassDefineOneShootingNodForCar("Car")
##sumsqr( Car_ODE.GetW() ) + sumsqr( Car_ODE.GetX() )
#ocp = OCP( "Optimize Simulated Car" ). \
#    AddDAE( Car_ODE ). \
#    AddLagrangeCostFunction( L = Car_ODE.GetX()[6]**2 ). \
#    AddLagrangeCostFunction( L = 1.0/Car_ODE.GetX()[2] ). \
#    AddMayerCostFunction( M = Mayer_exp ). \
#    SetStartTime( t0 = GroundTruthCar['t0'] ). \
#    SetEndTime( tf = tf ). \
#    SetX_0( x0 = GroundTruthCar['Sn'], mask=GroundTruthCar['S0_mask'] ). \
#    SetX_f( xf = GroundTruthCar['Xf'], mask=GroundTruthCar['Xf_mask'] ). \
#    SetLBW(lbw=GroundTruthCar['lbw'] ). \
#    SetUBW(ubw=GroundTruthCar['ubw'] ). \
#    SetLBQ(lbq=GroundTruthCar['lbq'] ). \
#    SetUBQ(ubq=GroundTruthCar['ubq'] ). \
#    SetNumberShootingNodes( Number = GroundTruthCar['NumberShootingNodes'] ). \
#    SetSolver( Solver = OptSolver.nlp )
##    Build()
#ocp.Build( GroundTruthCar )




#tf = SX.sym('tf',1)
#Mayer_exp = tf
#Car_ODE = ClassDefineOneShootingNodForCar("Car_Conf2")
##sumsqr( Car_ODE.GetW() ) + sumsqr( Car_ODE.GetX() )
#ocp = OCP( "Optimize Simulated Car_Conf2" ). \
#    AddDAE( Car_ODE ). \
#    AddLagrangeCostFunction( L = 0 ). \
#    AddMayerCostFunction( M = Mayer_exp ). \
#    SetStartTime( t0 = Car_Conf2['t0'] ). \
#    SetEndTime( tf = tf ). \
#    SetX_0( x0 = Car_Conf2['Sn'], mask=Car_Conf2['S0_mask'] ). \
#    SetX_f( xf = Car_Conf2['Xf'], mask=Car_Conf2['Xf_mask'] ). \
#    SetLBW(lbw=Car_Conf2['lbw'] ). \
#    SetUBW(ubw=Car_Conf2['ubw'] ). \
#    SetLBQ(lbq=Car_Conf2['lbq'] ). \
#    SetUBQ(ubq=Car_Conf2['ubq'] ). \
#    SetLB_State( eq=Car_Conf2['lb_state'] , mask=Car_Conf2['lb_state_mask'] ). \
#    SetUB_State( eq=Car_Conf2['ub_state'] , mask=Car_Conf2['ub_state_mask'] ). \
#    SetNumberShootingNodes( Number = Car_Conf2['NumberShootingNodes'] ). \
#    SetSolver( Solver = OptSolver.nlp )
##    Build()
#ocp.Build( Car_Conf2 )
