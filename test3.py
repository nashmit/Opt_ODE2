from casadi import *
import numpy as np
from enum import Enum


class OptSolver(Enum):
    nlp = 0
    qp = 1

def Pyridine(t0, tf):
    A = SX.sym('A')
    B = SX.sym('B')
    C = SX.sym('C')
    D = SX.sym('D')
    E = SX.sym('E')
    F = SX.sym('F')
    G = SX.sym('G')
    x = vertcat( A, B, C, D, E, F, G)
    #p = SX.sym('p',11)
    p1 = SX.sym('p1'); p2 = SX.sym('p2'); p3 = SX.sym('p3'); p4 = SX.sym('p4'); p5 = SX.sym('p5'); p6 = SX.sym('p6');
    p7 = SX.sym('p7'); p8 = SX.sym('p8'); p9 = SX.sym('p9'); p10 = SX.sym('p10'); p11 = SX.sym('p11');
    p = vertcat(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11);
    xdot = vertcat( -p1*A + p9*B,
                    p1*A - p2*B - p3*B*C - p7*D - p9*B + p10*D*F,
                    p2*B - p3*B*C - 2*p4*C*C -p6*C + p8*E + p10*D*F + 2*p11*E*F,
                    p3*B*C - p5*D - p7*D - p10*D*F,
                    p4*C*C +p5*D - p8*E - p11*E*F,
                    p3*B*C + p4*C*C + p6*C - p10*D*F - p11*E*F,
                    p6*C + p7*D + p8*F)
    ode = {'x':x, 'p':p, 'ode':xdot }
    I = integrator('I', 'cvodes', ode, {'t0': t0, 'tf': tf})

    return I, 7, 11
pass

def Notorious(t0, tf):
    x0 = SX.sym('x0')
    x1 = SX.sym('x1')
    x = vertcat (x0, x1)
    #mu = SX.sym('mu')
    mu = 60
    p = SX.sym('p')
    t = SX.sym('t')
    xdot = vertcat(x1, mu*mu*x0 - (mu*mu + p*p) * sin(p * t))
    ode = {'x':x, 't':t, 'p':p, 'ode':xdot}
    I = integrator('I','cvodes', ode, {'t0': t0, 'tf': tf})

    return I, 2, 1

GroundTruthNotorious = {
    'Sn': DM([0, pi]),
    'p': DM([pi])
}

GroundTruthNotorious_perturb = {
    'Sn': GroundTruthNotorious['Sn'] + 0.1 * np.random.normal(0, 0.4, GroundTruthNotorious['Sn'].shape),
    'p': GroundTruthNotorious['p'] + 0.1 * np.random.normal(0, 0.2, GroundTruthNotorious['p'].shape)
}

#Lotka–Volterra
def DefineOneShootingNodForDAE_LV(t0, tf):
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

    # returns [integrator][Nr equations][Nr parameters]
    return I, 2, 4;

GroundTruth = {
    'Sn': DM([20., 10.]),
    'p': DM([0.2, 0.01, 0.001, 0.1])
}

GroundTruth_perturb = {
    'Sn': GroundTruth['Sn'] + 1.1 * np.random.normal(0, 0.4, GroundTruth['Sn'].shape),
    'p': GroundTruth['p'] + 0.3 * np.random.normal(0, 0.02, GroundTruth['p'].shape)
}


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


GroundTruth = {
    'Sn': DM([20., 10.]),
    'p': DM([0.2, 0.01, 0.001, 0.1])
}

GroundTruth_perturb = {
    'Sn': GroundTruth['Sn'] + 1.1 * np.random.normal(0, 0.4, GroundTruth['Sn'].shape),
    'p': GroundTruth['p'] + 0.3 * np.random.normal(0, 0.02, GroundTruth['p'].shape)
}


def NLP(x0, p, eval_F1_exp, eval_F2_exp, GroundTruth_perturb, CasADi_norm):
    norm_F1_exp = CasADi_norm(eval_F1_exp(x0, p)) @ CasADi_norm(eval_F1_exp(x0, p))
    eval_norm_F1_exp = Function('eval_norm_F1_exp', [x0, p], [norm_F1_exp], ['x0_perturb', 'p_perturb'], ['out'])

    nlp = {'x': vertcat(reshape(x0, x0.numel(), 1), p),
           'f': eval_norm_F1_exp(x0, p),
           'g': eval_F2_exp(x0, p)
           }
    SolverNLP = nlpsol('S', 'ipopt', nlp)

    print(SolverNLP)

    initVal = GroundTruth_perturb
    # GroundTruth_perturb is used as starting point!
    result = SolverNLP(
        x0=vertcat(
            reshape(GroundTruth_perturb['Sn'], GroundTruth_perturb['Sn'].numel(), 1),
            GroundTruth_perturb['p']),
        lbg=0,
        ubg=0
        #lbg=repmat(0, x0.numel()-GroundTruth_perturb['Sn'].size1()),
        #ubg=repmat(0, x0.numel()-GroundTruth_perturb['Sn'].size1())
    )
    print('\n');
    print("x0_found: ", result['x'][:2], "p_found: ", result['x'][-p.numel():])

    print(initVal)
    print('\n')
    print(result['x'])
    print(GroundTruth)
    print("min:",result['f']);
    print('\n')
    pass


# add precision as a parameter!
def QP(x0, p, eval_F1_exp, eval_F2_exp, GroundTruth_perturb, CasADi_norm, precision_error = 0.001):
    # jacobian_exp = jacobian(F1_exp, vertcat(x0, p))
    jacobian_F1_exp = jacobian(eval_F1_exp(x0, p), vertcat(reshape(x0,x0.numel(),1), p))
    eval_jacobian_F1_exp = Function('eval_jacobian_exp', [x0, p], [jacobian_F1_exp], ['x0_perturb', 'p_perturb'], ['out'])

    DeltaX = MX.sym('DeltaX', x0.numel() + p.size1())

    inner_exp = eval_F1_exp(x0, p) + eval_jacobian_F1_exp(x0, p) @ DeltaX

    eval_inner_exp = Function('eval_inner_exp', [x0, p, DeltaX], [inner_exp], ['x0_perturb', 'p_perturb', 'DeltaX'],
                              ['out'])

    norm_GGN_exp = CasADi_norm(eval_inner_exp(x0, p, DeltaX)) @ CasADi_norm(eval_inner_exp(x0, p, DeltaX))

    # DeltaX must be DeltaX!
    eval_norm_GGN_exp = Function('eval_norm_GGN_exp', [x0, p, DeltaX], [norm_GGN_exp],
                                 ['x0_perturb', 'p_perturb', 'DeltaX'], ['out'])

    ##
    #eval F2(perturb)
    #compute J2()
    #evaluate F2 + J2 X DeltaX
    jacobian_F2_exp = jacobian(eval_F2_exp(x0, p), vertcat(reshape(x0,x0.numel(),1), p))
    eval_jacobian_F2_exp = \
        Function('eval_jacobian_F2_exp', [x0, p], [jacobian_F2_exp], ['x0_perturb', 'p_perturb'], ['out'])
    F2_inner_exp = eval_F2_exp(x0, p) + eval_jacobian_F2_exp(x0, p) @ DeltaX
    eval_F2_inner_exp = Function('eval_F2_inner_exp', [x0, p, DeltaX], [F2_inner_exp],
                                 ['x0_perturb', 'p_perturb', 'DeltaX'], ['out'])
    ##

    sol = vertcat(
        reshape(GroundTruth_perturb['Sn'], GroundTruth_perturb['Sn'].numel(), 1),
        GroundTruth_perturb['p']
    )
    shape = GroundTruth_perturb['Sn'].shape

    for idx in range(0, 20):
        # DeltaX must be DeltaX!!
        qp = {
            'x': DeltaX,
            'f': eval_norm_GGN_exp(reshape(sol[:x0.numel()],shape), sol[x0.numel(): ], DeltaX),
            'g': eval_F2_inner_exp(reshape(sol[:x0.numel()],shape), sol[x0.numel(): ], DeltaX)
        }
        SolverQP = qpsol('S', 'qpoases', qp)
        # print(SolverQP)
        result = SolverQP(x0=sol, lbg=0, ubg=0)
        print(sol)
        sol = sol + 0.1*result['x']
        print(sol)
        print("min:",result['f'])
        if (result['f']<=precision_error):
            print("Solution found!")
            break;

    pass


# trying to define identity function, polymorphically
# x0_aux = MX.sym('x0_aux',2,1)
# p_aux = MX.sym('p_aux',4,1)
# intg_exp_aux = MX.sym('intg_exp_aux')
# h_exp = Function('h_exp', [x0_aux, p_aux, intg_exp_aux], [intg_exp_aux], ['x0_aux', 'p_aux','intg_exp'], ['out'])


#NumberShootingNodes >=1
def ParametricEstimation(DAE_info, GroundTruth, GroundTruth_perturb, t, y, h_exp_input, Solver=OptSolver.nlp,
                         CasADi_norm=norm_2, t0=0, tf=5, NumberShootingNodes=10, TrueSolutionMeasurements=None):

    intg, SizeOfx0, SizeOfp = DAE_info(t0, tf / NumberShootingNodes)

    Sn, p, intg_exp = BuildCasADiExpFromIntegrator(NumberShootingNodes, intg=intg, SizeOfx0=SizeOfx0, SizeOfp=SizeOfp)

    if(TrueSolutionMeasurements==None):
        x0_aux, p_aux, intg_exp_for_Eta = \
            BuildCasADiExpFromIntegrator(NumberShootingNodes, intg=intg, SizeOfx0=SizeOfx0,
                                         SizeOfp=SizeOfp, mapaccum=True)
        eval_intg_exp_for_Eta = Function('eval_intg_exp_for_Eta', [x0_aux, p_aux], [intg_exp_for_Eta['xf']],
                                         ['x0', 'p'], ['out'])
        Eta_value = eval_intg_exp_for_Eta(GroundTruth['Sn'], GroundTruth['p'])
    else:
        Eta_value = TrueSolutionMeasurements
    #GroundTruth['Sn'] = horzcat(GroundTruth['Sn'], Eta_value)

    eval_intg_exp = Function('eval_intg_exp', [Sn, p], [intg_exp['xf']], ['x0', 'p'], ['out'])
    GroundTruth_perturb['Sn'] = horzcat(GroundTruth_perturb['Sn'], Eta_value[:, :-1])


    ######### trying to add polymorphism to generalize "h" function -> "dimensionality of "p" is the problem
    ######### in "eval_h_exp_input" when the "change of variable" is done.
    ######### need more investigation.
    eval_h_exp_input = Function('eval_h_exp_input', [t, y, p], [h_exp_input], ['t', 'y', 'p'], ['out'])
    t1 = MX.sym('t', Sn.size1(), NumberShootingNodes)
    y1 = MX.sym('y', Sn.size1(), NumberShootingNodes)
    h_exp_output = eval_h_exp_input(t1, y1, p)
    eval_h_exp_change_of_variable = Function('eval_h_exp_change_of_variable',
                                             [t, y, p], [h_exp_output], ['t', 'y', 'p'], ['out'])
    eval_h_exp_change_of_variable(
        repmat(np.arange(t0, tf, (tf-t0)/NumberShootingNodes), 1, Sn.size1()).T, intg_exp['xf'], p)
    ###########

    # this function is the identity function, but it can be much more and should be transmitted as a parameter
    # the polymorphic approach didn't work, must experiment more
    h_exp = Function('h_exp', [Sn, p], [intg_exp['xf']], ['Sn_perturb', 'p_perturb'], ['out'])

    # maybe I should express it as a function of Eta, will see
    # F1_exp = eval_intg_exp(GroundTruth['x0'], GroundTruth['p']) - h_exp(x0, p)
    # F1_exp = Eta_value - h_exp(x0, p,intg_exp['xf'])
    F1_exp = Eta_value - h_exp(Sn, p)
    F2_exp = eval_intg_exp(Sn, p)[:, :-1] - Sn[:, 1:]

    eval_F2_exp = Function('eval_F2_exp', [Sn, p], [reshape(F2_exp, F2_exp.numel(), 1)], ['Sn', 'p'], ['out']);

    # return in vector form after reshape!
    eval_F1_exp = Function('eval_F1_exp', [Sn, p], [reshape(F1_exp, F1_exp.numel(), 1)],
                           ['x0_perturb', 'p_perturb'], ['out'])

    if Solver == OptSolver.nlp:
        NLP(Sn, p, eval_F1_exp, eval_F2_exp, GroundTruth_perturb, CasADi_norm)

    if Solver == OptSolver.qp:
        QP(Sn, p, eval_F1_exp, eval_F2_exp, GroundTruth_perturb, CasADi_norm)

    pass


t = MX.sym('t')
y = MX.sym('y')
h_exp_input = t*y

#ParametricEstimation(DefineOneShootingNodForDAE_LV, GroundTruth, GroundTruth_perturb, OptSolver.qp)
ParametricEstimation(DefineOneShootingNodForDAE_LV, GroundTruth, GroundTruth_perturb,
                     t=t, y=y, h_exp_input=h_exp_input, Solver=OptSolver.nlp)

times = np.linspace(0, 1, 51, True)[1:]
t = MX.sym('t')
x0 = Function('x0',[t],[ sin( pi * t ) ],['t'],['out'])
x1 = Function('x1',[t],[ pi * cos( pi * t ) ], ['t'],['out'])
x = Function('x',[t],[ vertcat( x0(t), x1(t) ) ],['t'],['out'])
x_map = x.map(50)
TrueSolutionMeasurements = x_map(times)


#ParametricEstimation(Notorious, GroundTruthNotorious, GroundTruthNotorious_perturb, OptSolver.qp,
#                     norm_2, t0=0, tf=1, NumberShootingNodes=50)

