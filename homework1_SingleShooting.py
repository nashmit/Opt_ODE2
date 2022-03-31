from casadi import *
import numpy as np


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


intg, SizeOfx0, SizeOfp = DefineOneShootingNodForDAE_LV(t0=0, tf=0.1);


def BuildCasADiExpFromIntegrator(NumberShootingNodes, intg, SizeOfx0, SizeOfp):
    F = intg.mapaccum(NumberShootingNodes);
    x0 = MX.sym('x0', SizeOfx0);
    p = MX.sym('p', SizeOfp);
    intg_exp = F(x0=x0, p=repmat(p, 1, NumberShootingNodes));

    # return [symbolic input variables] and [symbolic integrator as expression]
    return x0, p, intg_exp;


x0, p, intg_exp = BuildCasADiExpFromIntegrator(NumberShootingNodes=20, intg=intg, SizeOfx0=SizeOfx0, SizeOfp=SizeOfp)

GroundTruth = {
    'x0': np.array([20., 10.]),
    'p': np.array([0.2, 0.01, 0.001, 0.1])
}

GroundTruth_perturb = {
    'x0': GroundTruth['x0'] + 1.1 * np.random.normal(0, 0.4, GroundTruth['x0'].shape),
    'p': GroundTruth['p'] + 0.3 * np.random.normal(0, 0.02, GroundTruth['p'].shape)
}

eval_intg_exp = Function('eval_intg_exp', [x0, p], [intg_exp['xf']], ['x0', 'p'], ['out'])
Eta_value = eval_intg_exp(GroundTruth['x0'], GroundTruth['p'])

# this function is identity, but it can be much more and is a parameter
h_exp = Function('h_exp', [x0, p], [intg_exp['xf']], ['x0_perturb', 'p_perturb'], ['out'])

# maybe I should express it as a function of Eta, will see
# F1_exp = eval_intg_exp(GroundTruth['x0'], GroundTruth['p']) - h_exp(x0, p)
F1_exp = Eta_value - h_exp(x0, p)

# return in vector form after reshape!
eval_F1_exp = Function('eval_F1_exp', [x0, p], [reshape(F1_exp, F1_exp.numel(), 1)],
                       ['x0_perturb', 'p_perturb'], ['out'])

# jacobian_exp = jacobian(F1_exp, vertcat(x0, p))
jacobian_exp = jacobian(eval_F1_exp(x0, p), vertcat(x0, p))
eval_jacobian_exp = Function('eval_jacobian_exp', [x0, p], [jacobian_exp], ['x0_perturb', 'p_perturb'], ['out'])

DeltaX = MX.sym('DeltaX', x0.size1() + p.size1())

inner_exp = eval_F1_exp(x0, p) + eval_jacobian_exp(x0, p) @ DeltaX

eval_inner_exp = Function('eval_inner_exp', [x0, p, DeltaX], [inner_exp], ['x0_perturb', 'p_perturb', 'DeltaX'],
                          ['out'])

norm_GGN_exp = norm_2(eval_inner_exp(x0, p, DeltaX)) @ norm_2(eval_inner_exp(x0, p, DeltaX))
# DeltaX must be DeltaX!
eval_norm_GGN_exp = Function('eval_norm_GGN_exp', [x0, p, DeltaX], [norm_GGN_exp],
                             ['x0_perturb', 'p_perturb', 'DeltaX'], ['out'])

norm_F1_exp = norm_2(eval_F1_exp(x0, p)) @ norm_2(eval_F1_exp(x0, p))
eval_norm_F1_exp = Function('eval_norm_F1_exp', [x0, p], [norm_F1_exp], ['x0_perturb', 'p_perturb'], ['out'])

nlp = {'x': vertcat(x0, p),
       'f': eval_norm_F1_exp(x0, p)}
SolverNLP = nlpsol('S', 'ipopt', nlp)
print(SolverNLP)
result = SolverNLP(x0=vertcat(GroundTruth_perturb['x0'], GroundTruth_perturb['p']))
print('\n');
print(result);
print('\n');

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
