from casadi import *
import numpy as np
from enum import Enum


class OptSolver(Enum):
    nlp = 0
    qp = 1


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

    # returns [integrator][Nr lines][Nr colums]
    return I, 2, 4;


GroundTruth = {
    'x0': np.array([20., 10.]),
    'p': np.array([0.2, 0.01, 0.001, 0.1])
}

GroundTruth_perturb = {
    'x0': GroundTruth['x0'] + 1.1 * np.random.normal(0, 0.4, GroundTruth['x0'].shape),
    'p': GroundTruth['p'] + 0.3 * np.random.normal(0, 0.02, GroundTruth['p'].shape)
}


def BuildCasADiExpFromIntegrator(NumberShootingNodes, intg, SizeOfx0, SizeOfp):
    F = intg.mapaccum(NumberShootingNodes);
    x0 = MX.sym('x0', SizeOfx0);
    p = MX.sym('p', SizeOfp);
    intg_exp = F(x0=x0, p=repmat(p, 1, NumberShootingNodes));

    # return [symbolic input variables] and [symbolic integrator as expression]
    return x0, p, intg_exp;


def NLP(x0, p, eval_F1_exp, GroundTruth_perturb, CasADi_norm):
    norm_F1_exp = CasADi_norm(eval_F1_exp(x0, p)) @ CasADi_norm(eval_F1_exp(x0, p))
    eval_norm_F1_exp = Function('eval_norm_F1_exp', [x0, p], [norm_F1_exp], ['x0_perturb', 'p_perturb'], ['out'])

    nlp = {'x': vertcat(x0, p),
           'f': eval_norm_F1_exp(x0, p)}
    SolverNLP = nlpsol('S', 'ipopt', nlp)

    print(SolverNLP)
    # GroundTruth_perturb is used as starting point!
    result = SolverNLP(x0=vertcat(GroundTruth_perturb['x0'], GroundTruth_perturb['p']))
    print('\n');
    print(result['x']);
    print(result['f']);
    print('\n')
    pass


# add precision as a parameter!
def QP(x0, p, eval_F1_exp, GroundTruth_perturb, CasADi_norm):
    # jacobian_exp = jacobian(F1_exp, vertcat(x0, p))
    jacobian_exp = jacobian(eval_F1_exp(x0, p), vertcat(x0, p))
    eval_jacobian_exp = Function('eval_jacobian_exp', [x0, p], [jacobian_exp], ['x0_perturb', 'p_perturb'], ['out'])

    DeltaX = MX.sym('DeltaX', x0.size1() + p.size1())

    inner_exp = eval_F1_exp(x0, p) + eval_jacobian_exp(x0, p) @ DeltaX

    eval_inner_exp = Function('eval_inner_exp', [x0, p, DeltaX], [inner_exp], ['x0_perturb', 'p_perturb', 'DeltaX'],
                              ['out'])

    norm_GGN_exp = CasADi_norm(eval_inner_exp(x0, p, DeltaX)) @ CasADi_norm(eval_inner_exp(x0, p, DeltaX))

    # DeltaX must be DeltaX!
    eval_norm_GGN_exp = Function('eval_norm_GGN_exp', [x0, p, DeltaX], [norm_GGN_exp],
                                 ['x0_perturb', 'p_perturb', 'DeltaX'], ['out'])

    sol = vertcat(GroundTruth_perturb['x0'], GroundTruth_perturb['p'])

    for idx in range(0, 10):
        # DeltaX must be DeltaX!!
        qp = {'x': DeltaX, 'f': eval_norm_GGN_exp(sol[0:x0.size1()], sol[x0.size1():sol.numel()], DeltaX)}
        SolverQP = qpsol('S', 'qpoases', qp)
        # print(SolverQP)
        result = SolverQP(x0=sol)
        print(sol)
        sol = sol + result['x']
        print(sol)
        print(result['f'])

    pass


# trying to define identity function, polymorphically
# x0_aux = MX.sym('x0_aux',2,1)
# p_aux = MX.sym('p_aux',4,1)
# intg_exp_aux = MX.sym('intg_exp_aux')
# h_exp = Function('h_exp', [x0_aux, p_aux, intg_exp_aux], [intg_exp_aux], ['x0_aux', 'p_aux','intg_exp'], ['out'])


def ParametricEstimation(DAE_info, GroundTruth, GroundTruth_perturb, Solver=OptSolver.nlp, CasADi_norm=norm_2, t0=0,
                         tf=0.1,
                         NumberShootingNodes=20):
    intg, SizeOfx0, SizeOfp = DAE_info(t0, tf)
    x0, p, intg_exp = BuildCasADiExpFromIntegrator(NumberShootingNodes, intg=intg, SizeOfx0=SizeOfx0, SizeOfp=SizeOfp)

    eval_intg_exp = Function('eval_intg_exp', [x0, p], [intg_exp['xf']], ['x0', 'p'], ['out'])
    Eta_value = eval_intg_exp(GroundTruth['x0'], GroundTruth['p'])

    # this function is the identity function, but it can be much more and should be transmitted as a parameter
    # the polymorphic approach didn't work, must experiment more
    h_exp = Function('h_exp', [x0, p], [intg_exp['xf']], ['x0_perturb', 'p_perturb'], ['out'])

    # maybe I should express it as a function of Eta, will see
    # F1_exp = eval_intg_exp(GroundTruth['x0'], GroundTruth['p']) - h_exp(x0, p)
    # F1_exp = Eta_value - h_exp(x0, p,intg_exp['xf'])
    F1_exp = Eta_value - h_exp(x0, p)

    # return in vector form after reshape!
    eval_F1_exp = Function('eval_F1_exp', [x0, p], [reshape(F1_exp, F1_exp.numel(), 1)],
                           ['x0_perturb', 'p_perturb'], ['out'])

    if Solver == OptSolver.nlp:
        NLP(x0, p, eval_F1_exp, GroundTruth_perturb, CasADi_norm)

    if Solver == OptSolver.qp:
        QP(x0, p, eval_F1_exp, GroundTruth_perturb, CasADi_norm)

    pass


ParametricEstimation(DefineOneShootingNodForDAE_LV, GroundTruth, GroundTruth_perturb, OptSolver.qp)

ParametricEstimation(DefineOneShootingNodForDAE_LV, GroundTruth, GroundTruth_perturb, OptSolver.nlp)
