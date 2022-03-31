from casadi import *
import numpy as np
from typing import Union

def BuildCasADiExpFromIntegrator(NumberShootingNodes, intg, SizeOfx0, SizeOfp, mapaccum=False):
    p = MX.sym('p', SizeOfp);

    if mapaccum == False:
        F = intg.map(NumberShootingNodes)
        Sn = MX.sym('Sn', SizeOfx0, NumberShootingNodes)
        intg_exp = F(x0=Sn, p=repmat(p, 1, NumberShootingNodes))
        # intg_exp = F(x0=Sn, p=p);  # f shootings + x0
    else:
        # don't forget to discard the first result as this one corresponds to x0 and we already have it
        F = intg.mapaccum(NumberShootingNodes)
        Sn = MX.sym('Sn', SizeOfx0)
        intg_exp = F(x0=Sn, p=repmat(p, 1, NumberShootingNodes))

    # return [symbolic input variables] and [symbolic integrator as expression]
    return Sn, p, intg_exp


class DAEInterface:

    def __init__(self, DAEName=''):
        self.DAEName = DAEName
        self.options = {}
        self.DefineDAE()


    def DefineDAE(self):
        assert False, "DefineDAE is not implemented!"
        pass

    def AddAlgebraicConstrain(self):
        assert False,"AddAlgebraicConstrain is not implemented!"
        pass

    def AddQuadrature(self, quad:Union[SX, MX]):
        self.dae['quad'] = quad
        pass

    def ResetCurrentQuadrature(self):
        self.dae['quad'] = 0

    def GetX(self):
        return self.x
        pass

    def GetP(self):
        return self.p
        pass

    def GetQ(self):
        #parameter
        return self.q
        pass

    def GetW(self):
        #control
        return self.w
        pass

    def GetXDot(self):
        return self.xdot
        pass

    def GetDAE(self):
        return self.dae
        pass

    def GetIntgType(self):
        return 'idas'
        pass

    def GetI(self, t0:float, tf:float):

        #assert False, "Don't use it!"

        I = integrator('I', self.GetIntgType(), self.dae, {'t0': t0, 'tf': tf} | self.options ) #, 'max_num_steps':200
        return I
        pass

    def GetI_TimeNormalized(self, t0, tf:Union[ SX, MX, float, int ], NumberShootingNodes:int ):
        # https://www.w3schools.com/python/ref_func_isinstance.asp

        if isinstance( tf, ( float, int ) ):
            assert tf>=t0, "tf must be bigger than t0!"

            if not hasattr(self, 'alreadyNormalized'):
                self.alreadyNormalized=True
                self.xdot = self.xdot * ( tf - t0 )
                # if it is a "true" DAE" this will break the current implementation
                # as it discard the algebraic part ... but, for this ex. it doesn't matter :)
                quad = None
                if 'quad' in self.dae:
                    quad = self.dae['quad'] * ( tf - t0 ) # re-normalize Lagrangian too!
                self.dae = { 'x':self.x, 'p':self.p, 'ode':self.xdot } # update "dae"
                if quad != None:
                    self.dae['quad'] = quad
            else:
                print("alreadyNormalized==True!")

            I = integrator('I', self.GetIntgType(), self.dae , {'t0': 0, 'tf': 1.0 / NumberShootingNodes } | self.options)
            #I = integrator('I', self.GetIntgType(), self.dae , {'t0': 0, 'tf': 1.0 / NumberShootingNodes } | self.options )
            return I
            pass

        if isinstance( tf, ( SX, MX ) ):
            if not hasattr(self, 'alreadyNormalized'):
                self.alreadyNormalized=True
                self.xdot = self.xdot * ( tf - t0 )
                self.q = vertcat( self.q, tf ) # add final time "tf" as parameter
                self.p = vertcat( self.w, self.q ) # update parameter structure
                # if it is a "true" DAE" this will break the current implementation
                # as it discard the algebraic part ... but, for this ex. it doesn't matter :)
                quad = None
                if 'quad' in self.dae:
                    quad = self.dae['quad'] * ( tf - t0 ) # re-normalize Lagrangian too!
                self.dae = { 'x':self.x, 'p':self.p, 'ode':self.xdot } # update "dae"
                if quad != None:
                    self.dae['quad'] = quad
            else:
                print("alreadyNormalized==True!")

            I = integrator('I', self.GetIntgType(), self.dae , {'t0': 0, 'tf': 1.0 / NumberShootingNodes } | self.options )
            #I = integrator('I', self.GetIntgType(), self.dae , {'t0': 0, 'tf': 1.0 / NumberShootingNodes } | self.options )
            return I
            #add time ("tf") as control parameter
            #t0 can be taken into consideration as some constant of the new 'xdot' so, don't add it as parameter!
            pass
        assert False, "Not the correct type of input parameter!"
        pass

    def GetI_plot(self, t0:float, tf:float, NrEvaluationPoints = 40):

        #assert False, "Don't use it!"

        time = np.insert(np.linspace(t0, tf, NrEvaluationPoints), t0, 0)
        I_plot = integrator('I_plot', self.GetIntgType(), self.dae, {'grid': time} | self.options) #, 'max_num_steps':200
        #I_plot = integrator('I_plot', self.GetIntgType(), self.dae, {'grid': time} | self.options ) #, 'max_num_steps':200
        return I_plot
        pass

    def GetI_plot_TimeNormalized(self, t0, tf:Union[ SX, MX, float, int ], NumberShootingNodes:int, NrEvaluationPoints):

        assert hasattr(self, 'alreadyNormalized'), "Must call: GetI_TimeNormalized(..) at least once before!"

        #if isinstance( tf, ( float, int ) ):
        time = np.insert(np.linspace(0, 1.0 / NumberShootingNodes, NrEvaluationPoints), t0, 0)
        I = integrator('I', self.GetIntgType(), self.dae , {'grid': time} | self.options )
        #I = integrator('I', self.GetIntgType(), self.dae , {'grid': time} | self.options )
        #no need to consider t0 and tf as parameters as they don't interfere in the optimization process!
        return I

        assert False, "This case should not not happen!"

        if isinstance( tf, ( SX, MX ) ):
            self.xdot = self.xdot * ( tf - t0 )
            self.q = vertcat( self.q, tf ) # add final time "tf" as parameter
            self.p = vertcat( self.w, self.q ) # update parameter structure
            # if it is a "true" DAE" this will break the current implementation
            # as it discard the algebraic part ... but, for this ex. it doesn't matter :)
            if 'quad' in self.dae:
                quad = self.dae['quad']
            self.dae = { 'x':self.x, 'p':self.p, 'ode':self.xdot } # update "dae"
            self.dae['quad'] = quad
            I = integrator('I', self.GetIntgType(), self.dae , {'t0': 0, 'tf': 1.0 / NumberShootingNodes } | self.options )
            if isinstance(quad, (SX, MX) ):
                self.dae['quad'] = quad
            #add time ("tf") as control parameter
            #t0 can be taken into consideration as some constant of the new 'xdot' so, don't add it as parameter!
            pass
        assert False, "Not the correct type of input parameter!"
        pass


class ClassDefineLotka_Volterra(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):

        x0 = SX.sym('x0')
        x1 = SX.sym('x1')
        x = vertcat(x0, x1)
        p00 = SX.sym('p00')
        p01 = SX.sym('p01')
        p10 = SX.sym('p10')
        p11 = SX.sym('p11')
        p4 = SX.sym('p4',1)
        p5 = SX.sym('p5',1)
        q = vertcat(p00, p01, p10, p11, p4, p5)
        w = SX.sym('w',1)
        #xdot = vertcat(p00 * x0 - p01 * x0 * x1 - p4 * x0 * w, p10 * x0 * x1 - p11 * x1 - p5 * x1 * w )
        xdot = vertcat(p00 * x0 - p01 * x0 * x1 - p4 * w, p10 * x0 * x1 - p11 * x1 - p5 * w )
        p = vertcat( w, q)
        ode = {'x': x, 'p': p, 'ode': xdot}

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals,
        self.xdot = xdot
        self.dae = ode


Config_LotkaVolterra= {
    't0': 0,
    'tf': 30,
    'NumberShootingNodes': 50,#50,
    'Sn': DM([20, 10]),
    'S0_mask': [1, 1],
    'SnName': [ 'x_0', 'x_1' ],
    'Xf': [90, 50],
    'Xf_mask': [1,0],#[0, 0],#[1, 1],
    'w': [ 0 ],
    'wName': ['u'],
    'lbw': [ -10 ],
    'ubw': [ 10 ],
    'q': [0.2, 0.01, 0.001, 0.1, 0.1, 0.1],
    'qName': [ 'p00', 'p01', 'p10', 'p11', 'p4', 'p5' ],
    'lbq': [0.2, 0.01, 0.001, 0.1, 0.1, 0.1 ],
    'ubq': [0.2, 0.01, 0.001, 0.1, 0.1, 0.1 ]
}


class ClassDefineOneShootingNodForVanDerPol(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):
        x0 = SX.sym('x0')
        x1 = SX.sym('x1')
        x = vertcat(x0, x1)
        k = SX.sym('k') # constant parameter
        #first is control and then parameter
        w = SX.sym('w') # control
        p = vertcat(w, k)
        xdot = vertcat(
            k * ( 1 - x1 * x1 ) * x0 - x1 + w,
            x0
        )
        ode = { 'x':x, 'p':p, 'ode':xdot }

        self.x = x
        self.p = p
        self.w = w # control
        self.q = k # constant parameter for all intervals, k must be 1 for this example!
        self.xdot = xdot
        self.dae = ode
        pass

# ODE definition etc. from: https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html
Config_VanDerPol= {
    't0': 0,
    'tf': 10,#20,
    'NumberShootingNodes': 5,#100,#4,
    'Sn': DM([0, 1]),#DM([1, 1]),
    'S0_mask': [1, 1],#[1, 1],
    'SnName': [ 'x_0', 'x_1' ],
    'Xf': [0.2, -2.2],#[0,0],#[ -1.5, -2.5 ],
    'Xf_mask': [1, 1],#[0, 1],#[1, 1],
    'w': [ -0.75 ],
    'wName': ['u'],
    'lbw': [-0.75],#[ -10.0 ],
    'ubw': [1.0],#[ 10.0 ],
    'q': [ 1 ],
    'qName': [ 'k' ],
    'lbq':[ 1 ],
    'ubq':[ 1 ]
}

class ClassDefineBrachistochrone(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):

        x = SX.sym('x',1)
        y = SX.sym('y',1)
        v = SX.sym('v',1)
        x = vertcat( x, y, v )
        w = SX.sym('w',1) # control, angle at each step
        p = vertcat(w)

        xdot = vertcat(
            v * sin( w ),
            v * cos( w ),
            9.81 * cos( w )
        )

        ode = { 'x':x, 'p':p,'ode':xdot }

        self.x = x
        self.p = p
        self.w = w # control
        self.q = vertcat() # no constant in this case!
        self.xdot = xdot
        self.dae = ode

        pass

Config_Brachistochrone = {
    't0': 0,
    'tf': 2, # used as the initiall guess and not as a fix final time!!!
    'tf_lb': 0, # automatically added in lbq
    'tf_ub':10, # automatically added in ubq (DON'T USE INF as you will end up with "INF" final expression which cannot be compared with other "INF" !!!!
    'NumberShootingNodes': 20,#4,
    'Sn': DM([0, 0, 0]),#DM([0, 0, -5]), # DM([0, 0, 0]),
    'S0_mask': [1, 1, 1],
    'SnName': [ 'x', 'y', 'v' ],
    'Xf': [10, 2, 0],
    'Xf_mask': [1, 1, 0],
    'w': [ 0 ],
    'wName': ['u'],
    'lbw': [-pi],#[ -10.0 ],
    'ubw': [pi],#[ 10.0 ],
    'q': [],
    'qName': [],
    'lbq':[],
    'ubq':[],
    'lb_state' :
        lambda state: Function(
            'lambda_lb_state', [ state ],
            [ vertcat( 0,  -0.5, 0 ) ], ['current_state'], ['lb_state'] ), # lower bound for 'v', without it, it start from 0
    'lb_state_mask': [ 0, 0, 0 ],
    'ub_state' :
        lambda state: Function(
            'lambda_ub_state',  [ state ],
            [ vertcat( 0, 2.5, 0 ) ], ['current_state'], ['ub_state'] ), # y must be at most 2.5 ( without it, optimal path touches ~3.0 )
    'ub_state_mask': [ 0, 0, 0 ]
}



class ClassDefineOneShootingNodForCar(DAEInterface):

    def __init__(self, DAEName="DefaultCar"):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):

        # state variables
        # position
        c_x = SX.sym('c_x', 1)
        c_y = SX.sym('c_y', 1)
        # magnitude of directional velocity of the car
        v = SX.sym('v', 1)
        # steering wheel angle
        delta = SX.sym('delta', 1)
        # side slip angle
        beta = SX.sym('beta', 1)
        # yaw angle
        psi = SX.sym('psi', 1)
        # yaw angle velocity
        omega_z = SX.sym('omega_z', 1)

        # state space
        x = vertcat( c_x, c_y, v, delta, beta, psi, omega_z )

        ### control parameters!
        # steering wheel angular velocity
        omega_delta = SX.sym('omega_delta', 1)
        # total braking force controlled by the driver
        F_B = SX.sym('F_B', 1)
        # accelerator pedal position
        phi = SX.sym('phi', 1)
        # this control parameter is considered as a constant one and not controllable one!
        # selected gear
        mu = SX.sym('mu', 1)

        # concatenate control parameters, without "mu"
        w = vertcat(omega_delta, F_B, phi) #, mu)

        ### constant parameters!
        # dist. centre of gravity to drag mount point
        e_SP = SX.sym('e_SP', 1)

        # moment of inertia
        I_zz = SX.sym('I_zz', 1)

        #wheel radius
        R = SX.sym('R', 1)

        #motor torque transmission
        i_t = SX.sym('i_t', 1)

        # mass
        m = SX.sym('m', 1)

        # acceleration due to gravity
        g = SX.sym('g', 1)

        # dist. centre of gravity to 'front' / 'rear' wheel
        l_f = SX.sym('l_f', 1)
        l_r = SX.sym('l_r', 1)

        # coefficients
        f_R0 = SX.sym('f_R0', 1)
        f_R1 = SX.sym('f_R1', 1)
        f_R4 = SX.sym('f_R4', 1)

        # air drag coefficient
        c_omega = SX.sym('c_omega', 1)

        # air density
        rho = SX.sym('rho', 1)

        # effective flow surface
        A = SX.sym('A', 1)

        # Pacejka-model ( stiffness factor )
        B_f = SX.sym('B_f', 1)
        B_r = SX.sym('B_r', 1)
        # Pacejka-model ( shape factor ): C_f == C_r
        C_f = SX.sym('C_f', 1)
        C_r = SX.sym('C_r', 1)
        # Pacejka-model ( peak value )
        D_f = SX.sym('D_f', 1)
        D_r = SX.sym('D_r', 1)
        # Pacejka-model ( curvature factor ): E_f == E_r
        E_f = SX.sym('E_f', 1)
        E_r = SX.sym('E_r', 1)

        # concatenate constant parameters
        q = vertcat(
            m,
            g,
            l_f, l_r,
            e_SP,
            R,
            I_zz,
            c_omega,
            rho,
            A,
            mu,
            i_t,
            B_f, B_r,
            C_f, C_r,
            D_f, D_r,
            E_f, E_r,
            f_R0, f_R1, f_R4
        )

        # concatenate all parameters
        p = vertcat(w, q)

        # slip angles are given by:
        #'omega_z' is 'psi_dot' I'm using directly 'omega_z' to fix the recursive definition !!!
        #alpha_f = delta - atan2( l_f * psi_dot - v * sin( beta ), v * cos( beta ) )
        #alpha_r = atan2( l_f * psi_dot + v * sin( beta ), v * cos( beta ) )
        alpha_f = delta - atan2( l_f * omega_z - v * sin( beta ), v * cos( beta ) )
        alpha_r = atan2( l_f * omega_z + v * sin( beta ), v * cos( beta ) )

        # 'lateral tire forces' are functions of respective 'slip angles' ( and the 'tire loads', which are
        # constant in out model )
        #F_sf = Function('F_sf', [ alpha_f ],
        #                [ D_f * sin( C_f * atan( B_f * alpha_f - E_f * ( B_f * alpha_f - atan( B_f * alpha_f )))) ],
        #                ['alpha_f'], ['out'] )
        F_sf = D_f * sin( C_f * atan( B_f * alpha_f - E_f * ( B_f * alpha_f - atan( B_f * alpha_f ))))
        #F_sr = Function('F_sr', [ alpha_r ],
        #                [ D_r * sin( C_r * atan( B_r * alpha_r - E_r * ( B_r * alpha_r - atan( B_r * alpha_r )))) ],
        #                ['alpha_r'], ['out'] )
        F_sr = D_r * sin( C_r * atan( B_r * alpha_r - E_r * ( B_r * alpha_r - atan( B_r * alpha_r ))))


        # drag due to air resistance
        F_Ax = 1 / 2 * c_omega * rho * A * v*v
        F_Ay = 0 # no side wind

        # some function useful for computing 'rolling resistance force'
        f_R = Function('f_R', [ v ], [ f_R0 + f_R1 * v / 100 + f_R4 * ( v / 100 )**4 ] )

        #friction coefficients:
        F_zf = ( m * l_r * g ) / ( l_f + l_r )
        F_zr = ( m * l_f * g ) / ( l_f + l_r )

        # rolling resistance force
        F_Rf = f_R( v ) * F_zf
        F_Rr = f_R( v ) * F_zr

        # front wheels braking force
        F_Bf = 2/3 * F_B
        # rear wheels braking force
        F_Br = 1/3 * F_B

        # longitudinal tire force at the front wheel
        F_lf = - F_Bf - F_Rf

        #gear selection 1..5
        i_g_4_5 = Function('i_g_4_5', [mu], [ if_else( mu==4, 1.0, 0.805) ] )
        i_g_3 = Function('i_g_3', [mu], [ if_else( mu==3, 1.33, i_g_4_5( mu ) ) ] )
        i_g_2 = Function('i_g_2', [mu], [ if_else( mu==2, 2.002, i_g_3( mu ) ) ] )
        i_g = Function('i_g', [mu], [ if_else( mu==1, 3.91, i_g_2(mu) ) ], ['mu'], ['out'] )

        # rotary frequency of the motor depending on the gear 'mu'
        #omega_mot = Function('omega_mot', [ mu ], [ v * i_g( mu ) * i_t / R ], ['omega_mot'], ['out'])
        omega_mot = v * i_g( mu ) * i_t / R

        #f1 = Function('f1', [ phi ], [ 1 - exp( -3 * phi ) ], [ 'phi' ], ['out'] )
        f1 = 1 - exp( -3 * phi )
        #f2 = Function('f2', [ omega_mot ],
        #              [ -37.8 + 1.54 * omega_mot - 0.0019 * omega_mot * omega_mot ],
        #              ['omega_mot'], ['out'] )
        f2 = -37.8 + 1.54 * omega_mot - 0.0019 * omega_mot * omega_mot
        #f3 = Function('f3', [ omega_mot ], [ - 34.9 - 0.04775 * omega_mot ], ['omega_mot'], ['out'] )
        f3 = - 34.9 - 0.04775 * omega_mot

        #motor torque
        #M_mot = Function('M_mot', [ phi, mu ],
        #                 [ f1( phi ) * f2( omega_mot( mu ) ) + ( 1 - f1( phi ) ) * f3( omega_mot( mu ) ) ],
        #                 ['phi', 'mu'], ['out'])
        M_mot = f1 * f2 + ( 1 - f1 ) * f3


        #torque resulting from drive-train applied to the rear wheel
        #M_wheel = Function('M_wheel',[ phi, mu ],[ i_g( mu ) * i_t * M_mot( phi, mu ) ], ['phi', 'mu'],['out'])
        M_wheel = Function('M_wheel',[ phi, mu ],[ i_g( mu ) * i_t * M_mot ], ['phi', 'mu'],['out'])


        # Longitudinal tire force at the rear wheel
        F_lr = M_wheel( phi, mu ) / R - F_Br - F_Rr

        c_x_dot = v * cos( psi - beta ) #done!

        c_y_dot = v * sin( psi - beta ) #done!

        # the "...( F_lr ** mu - F_Ax )..." in the main article power "mu" is "1"
        v_dot = 1 / m * ( ( F_lr - F_Ax ) * cos( beta ) + \
                F_lf * cos( delta + beta ) - \
                ( F_sr - F_Ay ) * sin( beta ) - \
                F_sf * sin( delta + beta ) )

        delta_dot = omega_delta

        beta_dot = omega_z - \
                   1 / ( m * v ) * ( ( F_lr - F_Ax ) * sin( beta ) + F_lf * sin( delta + beta ) + \
                                     ( F_sr - F_Ay ) * cos( beta ) + F_sf * cos( delta + beta ) )

        psi_dot = omega_z

        omega_z_dot = 1 / I_zz * ( F_sf * l_f * cos( delta ) - F_sr * l_r - F_Ay * e_SP + F_lf * l_f * sin( delta ) )

        xdot = vertcat(
            c_x_dot,
            c_y_dot,
            v_dot,
            delta_dot,
            beta_dot,
            psi_dot,
            omega_z_dot
        )

        ode = { 'x':x, 'p':p, 'ode':xdot }

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.xdot = xdot
        self.dae = ode

        pass


#from test18_if_else_implementation import P_l_Function, P_u_Function
GroundTruthCar= {
    't0': 0,
    'tf': 4,
    'tf_lb': 0,
    'tf_ub':10,
    'car_width': 1.5,
    'NumberShootingNodes': 15,
    'Sn': DM([0, 0, 3, 0, 0, 0, 0]),#DM([-30, 0, 10, 0, 0, 0, 0]),
    'S0_mask': [1, 1, 1, 1, 1, 1, 1],
    'SnName': ['c_x', 'c_y', 'v', 'delta', 'beta', 'psi', 'omega_z'],
    'Xf': [4, 2, 0, 0, 0, 0, 0],#[140, 0, 0, 0, 0, 0, 0],
    'Xf_mask': [1, 1, 0, 0, 0, 0, 0],#[1, 0, 0, 0, 0, 1, 0],
    'w': [ 0.2, 0, 0.22 ],
    'lbw': [ -0.5, 0, 0 ],
    'ubw': [ 0.5, 1.5*10**4, 1 ],
    'wName': ['omega_delta', 'F_B', 'phi'], #'mu'],
    'q': [ 1239, 9.81, 1.19016, 1.37484, 0.5, 0.302, 1.752, 0.3, 1.249512, 1.4378946874, 2, 3.91,
           10.96, 12.67, 1.3, 1.3, 4560.40, 3947.81, -0.5, -0.5, 0.009, 0.002, 0.0003 ],
    'qName': [ 'm', 'g', 'l_f', 'l_r', 'e_SP', 'R', 'I_zz', 'c_omega', 'rho', 'A', 'mu', 'i_t',
               'B_f', 'B_r', 'C_f', 'C_r', 'D_f', 'D_r', 'E_f', 'E_r', 'f_R0', 'f_R1', 'f_R4' ],
    'lbq':[ 1239, 9.81, 1.19016, 1.37484, 0.5, 0.302, 1.752, 0.3, 1.249512, 1.4378946874, 2, 3.91,
            10.96, 12.67, 1.3, 1.3, 4560.40, 3947.81, -0.5, -0.5, 0.009, 0.002, 0.0003 ],
    'ubq':[ 1239, 9.81, 1.19016, 1.37484, 0.5, 0.302, 1.752, 0.3, 1.249512, 1.4378946874, 2, 3.91,
            10.96, 12.67, 1.3, 1.3, 4560.40, 3947.81, -0.5, -0.5, 0.009, 0.002, 0.0003 ],
    'lb_state' :
        lambda state: Function(
            'lambda_lb_state', [ state ],
            [ vertcat( 0, P_l_Function( state[0] ), 0, 0, 0, 0, 0 ) ], ['current_state'], ['lb_state'] ), # lower bound for 'y' and velocity norm
    'lb_state_mask': [ 0, 1, 0, 0, 0, 0, 0 ], # lower bound for 'y' and velocity norm
    'ub_state' :
        lambda state: Function(
            'lambda_ub_state',  [ state ],
            [ vertcat( 0, P_u_Function( state[0] ),   0, 0, 0, 0, 0 ) ], ['current_state'], ['ub_state'] ),
    'ub_state_mask': [ 0, 1, 0, 0, 0, 0, 0 ]
}
#GroundTruthCar['p'] = GroundTruthCar['w'] + GroundTruthCar['q']



Car_Conf2= {
    't0': 0,
    'tf': 5,
    'tf_lb': 0,
    'tf_ub':20,
    'car_width': 1.5,
    'NumberShootingNodes': 30,
    'Sn': DM([-30, 0, 10, 0, 0, 0, 0]),#DM([-30, 0, 10, 0, 0, 0, 0]),
    'S0_mask': [1, 1, 1, 1, 1, 1, 1],
    'SnName': ['c_x', 'c_y', 'v', 'delta', 'beta', 'psi', 'omega_z'],
    'Xf': [140, 0, 0, 0, 0, 0, 0],
    'Xf_mask': [1, 0, 0, 0, 0, 1, 0],
    'w': [ 0.2, 0, 0.22 ],
    'lbw': [ -0.5, 0, 0 ],
    'ubw': [ 0.5, 1.5*10**4, 1 ],
    'wName': ['omega_delta', 'F_B', 'phi'], #'mu'],
    'q': [ 1239, 9.81, 1.19016, 1.37484, 0.5, 0.302, 1.752, 0.3, 1.249512, 1.4378946874, 2, 3.91,
           10.96, 12.67, 1.3, 1.3, 4560.40, 3947.81, -0.5, -0.5, 0.009, 0.002, 0.0003 ],
    'qName': [ 'm', 'g', 'l_f', 'l_r', 'e_SP', 'R', 'I_zz', 'c_omega', 'rho', 'A', 'mu', 'i_t',
               'B_f', 'B_r', 'C_f', 'C_r', 'D_f', 'D_r', 'E_f', 'E_r', 'f_R0', 'f_R1', 'f_R4' ],
    'lbq':[ 1239, 9.81, 1.19016, 1.37484, 0.5, 0.302, 1.752, 0.3, 1.249512, 1.4378946874, 2, 3.91,
            10.96, 12.67, 1.3, 1.3, 4560.40, 3947.81, -0.5, -0.5, 0.009, 0.002, 0.0003 ],
    'ubq':[ 1239, 9.81, 1.19016, 1.37484, 0.5, 0.302, 1.752, 0.3, 1.249512, 1.4378946874, 2, 3.91,
            10.96, 12.67, 1.3, 1.3, 4560.40, 3947.81, -0.5, -0.5, 0.009, 0.002, 0.0003 ],
    'lb_state' :
        lambda state: Function(
            'lambda_lb_state', [ state ],
            [ vertcat( 0, P_l_Function( state[0] ), 3, 0, 0, 0, 0 ) ], ['current_state'], ['lb_state'] ), # lower bound for 'y' and velocity norm
    'lb_state_mask': [ 0, 1, 0, 0, 0, 0, 0 ], # lower bound for 'y' and velocity norm
    'ub_state' :
        lambda state: Function(
            'lambda_ub_state', [ state ],
            [ vertcat( 0, P_u_Function( state[0] ), 0, 0, 0, 0, 0 ) ], ['current_state'], ['ub_state'] ),
    'ub_state_mask': [ 0, 1, 0, 0, 0, 0, 0 ]
}
#Car_Conf2['p'] = Car_Conf2['w'] + Car_Conf2['q']






class ClassDefineRocket_Car(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):

        s = MX.sym('s',1) #position
        v = MX.sym('v',1) #velocity e.g. v = ds/dt

        x = vertcat(s, v) # state space

        q = vertcat()
        w = MX.sym('w',1) # control
        p = vertcat( w, q)

        xdot = DM( [[0, 1], [0, 0]] ) @ x + DM( [0, 1] ) * w
        ode = {'x': x, 'p': p, 'ode': xdot}

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.xdot = xdot
        self.dae = ode


Config_Rocket_Car= {
    't0': 0,
    'tf': 32,
    'tf_lb': 0,
    'tf_ub':32,
    'NumberShootingNodes': 10,#50,
    'Sn': DM([0, 0]), # position and velocity at t_0
    'S0_mask': [1, 1],
    'SnName': [ 's', 'v' ],
    'Xf': [300, 0], # position and velocity at tf
    'Xf_mask': [1,1],#[0, 0],#[1, 1],
    'w': [ 0.3 ],
    'wName': ['u'],
    'lbw': [ -2 ],
    'ubw': [ 1 ],
    'q': [],
    'qName': [],
    'lbq': [],
    'ubq': []
}


class ClassDefineInvertedPendulumOnACard(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):

        r = SX.sym( 'r', 1 )
        rDot = SX.sym( 'rDot', 1 )
        theta = SX.sym( 'theta', 1 )
        thetaDot = SX.sym( 'thetaDot', 1 )
        x = vertcat( r, rDot, theta, thetaDot ) # state space

        m = SX.sym( 'm',1 ) # link mass constant parameter
        M = SX.sym( 'M',1 ) # cart mass constant parameter
        g = SX.sym( 'g',1 ) # gravitational acceleration constant parameter
        l = SX.sym( 'l',1 ) # link length constant parameter
        q = vertcat( m, M, g, l ) # fixed constant parameters for all shooting nodes

        u_r = SX.sym( 'u_r', 1 ) # control parameter
        w = vertcat( u_r ) # vector of control parameter

        p = vertcat( w, q ) # complete vector of parameters ( control param. and constant param. )

        xdot = vertcat(
            rDot,
            1.0 / ( M + m * sin( theta )**2 ) * ( -m * g * cos( theta ) * sin( theta ) - m * l * thetaDot**2 * sin( theta ) + u_r ),
            thetaDot,
            g / l * sin( theta ) +
            (1.0 / ( M + m * sin( theta )**2 ) * ( -m * g * cos( theta ) * sin( theta ) - m * l * thetaDot**2 * sin( theta ) + u_r )) / l * cos( theta )
        )

        ode = {'x': x, 'p': p, 'ode': xdot}

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.xdot = xdot
        self.dae = ode



Config_InvertedPendulumOnACard= {
    't0': 0,
    'tf': 10,
    'tf_lb': 1,
    'tf_ub':100,
    'NumberShootingNodes': 20,#50,
    'Sn': DM([0, 0, 0, 0]), # position and velocity at t_0
    'S0_mask': [1, 1, 1, 1],
    #'S0_lb': [0, 0, 0, 0],
    #'S0_ub': [0, 0, 0, 0],
    'SnName': [ 'r', 'rDot', 'theta', 'thetaDot' ],
    #'Xf': [0, 0, pi, 0], # 'position' and 'velocity' at tf of each DOF
    #'Xf_mask': [0, 0, 1, 1],#[0, 0],#[1, 1], ->final velocity is not that important
    'Xf_lb': lambda state: Function(
        'lambda_Xf_lb_state',
        [ state ], [ vertcat( 0, 0, pi-pi/32, 0 ) ],
        ['current_state'], ['Xf_lb'] ), # lower bound for 'theta' and 'thetaDot'
    'Xf_lb_mask': [0, 0, 1, 1],
    'Xf_ub': lambda state: Function(
        'lambda_Xf_ub_state',
        [ state ], [ vertcat( 0, 0, pi-pi/32, 0 ) ],
        ['current_state'], ['Xf_ub'] ), # upper bound for 'theta' and 'thetaDot'
    'Xf_ub_mask': [0, 0, 1, 1],
    'w': [ 0 ], # doesn't influence the ODE
    'wName': ['u_r'], #control
    'lbw': [ -20 ],#[-20],
    'ubw': [  20 ],#[20],
    'q': [1, 10, -9.8, 1.5],
    'qName': ['m','M','g','l'],
    'lbq': [1, 10, -9.8, 1.5],
    'ubq': [1, 10, -9.8, 1.5],
    'cart_width': 0.5,
    'cart_height': 0.3,
    'lb_state' :
        lambda state: Function(
            'lambda_lb_state',
            [ state ], [ vertcat( -0.7, -2.2, 0, 0 ) ],
            ['current_state'], ['lb_state'] ), # lower bound for 'r' and velocity norm
    'lb_state_mask': [ 1, 1, 0, 0 ], # lower bound mask for 'r' and velocity norm
    'ub_state' :
        lambda state: Function(
            'lambda_ub_state',
            [ state ], [ vertcat(  0.7,  2.2, 0, 0 ) ],
            ['current_state'], ['ub_state'] ), # upper bound for 'r' and velocity norm
    'ub_state_mask': [ 1, 1, 0, 0 ], # upper bound mask for 'r' and velocity norm

    'lb_state_phase2' :
        lambda state: Function(
            'lambda_lb_state',
            [ state ], [ vertcat( -0.7, -2.2, pi-pi/32, 0 ) ],
            ['current_state'], ['lb_state'] ), # lower bound for 'r' and velocity norm
    'lb_state_mask_phase2': [ 1, 1, 1, 0 ], # lower bound mask for 'r' and velocity norm
    'ub_state_phase2' :
        lambda state: Function(
            'lambda_ub_state',
            [ state ], [ vertcat(  0.7,  2.2, pi+pi/32, 0 ) ],
            ['current_state'], ['ub_state'] ), # upper bound for 'r' and velocity norm
    'ub_state_mask_phase2': [ 1, 1, 1, 0 ], # upper bound mask for 'r' and velocity norm
}


Config_InvertedPendulumOnACard_NMPC= {
    't0': 0,
    'tf': 2,
    'tf_lb': 1,
    'tf_ub': 5,
    'NumberShootingNodes': 20,#50,
    'Sn': DM([0, 0, 0, 0]), # position and velocity at t_0
    'S0_mask': [1, 1, 1, 1],
    #'S0_lb': [0, 0, 0, 0],
    #'S0_ub': [0, 0, 0, 0],
    'SnName': [ 'r', 'rDot', 'theta', 'thetaDot' ],
    #'Xf': [0, 0, pi, 0], # 'position' and 'velocity' at tf of each DOF
    #'Xf_mask': [0, 0, 1, 1],#[0, 0],#[1, 1], ->final velocity is not that important
    'Xf_lb': lambda state: Function(
        'lambda_Xf_lb_state',
        [ state ], [ vertcat( 0.5, 0, pi, 0 ) ],
        ['current_state'], ['Xf_lb'] ), # lower bound for 'theta' and 'thetaDot'
    'Xf_lb_mask': [1, 0, 1, 1],
    'Xf_ub': lambda state: Function(
        'lambda_Xf_ub_state',
        [ state ], [ vertcat( 0.5, 0, pi, 0 ) ],
        ['current_state'], ['Xf_ub'] ), # upper bound for 'theta' and 'thetaDot'
    'Xf_ub_mask': [1, 0, 1, 1],
    'w': [ 0 ], # doesn't influence the ODE
    'wName': ['u_r'], #control
    'lbw': [ -20 ],#[-20],
    'ubw': [  20 ],#[20],
    'q': [1, 10, -9.8, 1.5],
    'qName': ['m','M','g','l'],
    'lbq': [1, 10, -9.8, 1.5],
    'ubq': [1, 10, -9.8, 1.5],
    'cart_width': 0.5,
    'cart_height': 0.3,
    'lb_state' :
        lambda state: Function(
            'lambda_lb_state',
            [ state ], [ vertcat( -0.7, -2.2, 0, 0 ) ],
            ['current_state'], ['lb_state'] ), # lower bound for 'r' and velocity norm
    'lb_state_mask': [ 1, 1, 0, 0 ], # lower bound mask for 'r' and velocity norm
    'ub_state' :
        lambda state: Function(
            'lambda_ub_state',
            [ state ], [ vertcat(  0.7,  2.2, 0, 0 ) ],
            ['current_state'], ['ub_state'] ), # upper bound for 'r' and velocity norm
    'ub_state_mask': [ 1, 1, 0, 0 ], # upper bound mask for 'r' and velocity norm
}




class ClassDefineBall(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):

        s = SX.sym('s',1) #postion
        v = SX.sym('v',1) #velocity e.g. v = ds/dt

        x = vertcat(s, v) # state space

        g = SX.sym('g',1) # gravitational acceleration
        q = vertcat(g) # parameter, same for all intervals, it can have lb<ub in this ex. lb=ub
        acc = SX.sym('acc',1) # extra acceleration / control parameter
        w = vertcat(acc) # control ( will be 0 ) lb<=0<=ub ( extra acceleration )
        p = vertcat( w, q)

        xdot = vertcat(v,g)
        ode = {'x': x, 'p': p, 'ode': xdot}

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.xdot = xdot
        self.dae = ode


Config_Ball= {
    't0': 0,
    'tf': 10,
    'tf_lb': 0,
    'tf_ub':100,
    'NumberShootingNodes': 10,#50,
    'Sn': DM([100, 0]), # position and velocity at t_0
    'S0_mask': [1, 1],
    'SnName': [ 's', 'v' ],
    'Xf': [0, 0], # position and velocity at tf
    'Xf_mask': [1, 0],#[0, 0],#[1, 1], ->final velocity is not inport as we are searching for the time of impact
    'w': [0], # doesn't influence the ODE
    'wName': ['acc'], #control
    'lbw': [0],
    'ubw': [0],
    'q': [ -9.8 ],
    'qName': ['g'],
    'lbq': [-9.8],
    'ubq': [-9.8]
}



class ClassDefinePendulum(DAEInterface):

    def __init__(self, DAEName=""):
        super().__init__(DAEName=DAEName)
        pass

    def DefineDAE(self):

        theta = SX.sym('theta', 1)
        thetaDot = SX.sym('thetaDot', 1)

        x = vertcat(theta, thetaDot) # state space

        g = SX.sym( 'g', 1 ) # gravitational acceleration
        l = SX.sym( 'l', 1 ) # rod lenght
        q = vertcat( g, l ) # parameter, same for all intervals, it can have lb<ub in this ex. lb=ub
        acc = SX.sym( 'acc', 1 ) # extra acceleration / control parameter
        w = vertcat( acc ) # control ( will be 0 ) lb<=0<=ub ( extra acceleration )
        p = vertcat( w, q )

        xdot = vertcat(
            thetaDot,
            -g / l * sin( theta  ) + acc
        )
        ode = {'x': x, 'p': p, 'ode': xdot}

        self.x = x
        self.p = p
        self.w = w # control
        self.q = q # constant parameter for all intervals
        self.xdot = xdot
        self.dae = ode

Config_Pendulum= {
    't0': 0,
    'tf': 2,
    'tf_lb': 0,
    'tf_ub':50,
    'NumberShootingNodes': 10,#50,
    'Sn': DM([pi/6, 0]), # position and velocity at t_0
    'S0_mask': [1, 1],
    'SnName': [ 'theta', 'thetaDot' ],
    'Xf': [pi, 0], # position and velocity at tf
    'Xf_mask': [1, 0],#[0, 0],#[1, 1], ->final velocity is not inport as we are searching for the time of impact
    'w': [0], # doesn't influence the ODE
    'wName': ['acc'], #control
    'lbw': [-1],
    'ubw': [1],
    'q': [ -9.8 , 0.5 ],
    'qName': ['g', 'l'],
    'lbq': [-9.8, 0.5 ],
    'ubq': [-9.8, 0.5 ]
}