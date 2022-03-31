from casadi import *
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum


#Lotka–Volterra
def DefineOneShootingNodForDAE_LV(t0, tf, NrEvaluationPoints = 40):
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

    time = np.insert(np.linspace(t0, tf, NrEvaluationPoints), t0, 0)
    I = integrator('I', 'cvodes', ode, {'t0': t0, 'tf': tf});
    I_plot = integrator('I_plot', 'cvodes', ode, {'grid': time});

    # returns [integrator][Nr equations][Nr parameters]
    return I, 2, 4, I_plot;

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

t0 = 0
tf = 200
NumberShootingNodes = 10

class ShootingPlot:
    def __init__(self, t0, tf, NumberShootingNodes, DAE_info, NrEvaluationPoints=40):

        self.t0 = t0
        self.tf = tf
        self.NumberShootingNodes = NumberShootingNodes

        self.NrEvaluationPoints = NrEvaluationPoints

        intg, SizeOfx0, SizeOfp, I_plot =\
            DAE_info(t0, (tf-t0)/NumberShootingNodes, NrEvaluationPoints=NrEvaluationPoints)

        self.Sn, self.p, self.intg_exp = \
            BuildCasADiExpFromIntegrator(NumberShootingNodes, I_plot, SizeOfx0, SizeOfp, False)
        self.eval_intg_exp = Function('eval_intg_exp', [self.Sn, self.p], [self.intg_exp['xf']], ['x0', 'p'], ['out'])

        self.Sn_mapaccum, self.p_mapaccum, self.intg_exp_mapaccum = \
            BuildCasADiExpFromIntegrator(NumberShootingNodes, I_plot, SizeOfx0, SizeOfp, True)
        self.eval_intg_exp_mapaccum = \
            Function('eval_intg_exp_mapaccum',
                     [self.Sn_mapaccum, self.p_mapaccum], [self.intg_exp_mapaccum['xf']], ['x0', 'p'], ['out'])
    pass

    def Plot(self, x0, p, mapaccum=False):
        # https://stackoverflow.com/questions/13359951/is-there-a-list-of-line-styles-in-matplotlib
        linestyle = {0: '-', 1: '-.', 2: '--', 3: ':'}

        if mapaccum == False:
            res = self.eval_intg_exp(x0, p)
        else:
            res = self.eval_intg_exp_mapaccum(x0, p)


        for dof in range(0, len(x0)):
            indx_shoots = 0
            legend_ok=True
            for indx in np.arange(self.t0, self.tf, (self.tf-self.t0)/self.NumberShootingNodes):
                if legend_ok:
                    label=str(dof) + ' dof'
                else:
                    label=''
                res_x_dof = res[dof, :]
                shoots_x_dof = reshape(res_x_dof, self.NrEvaluationPoints, self.NumberShootingNodes)
                plt.plot(np.linspace(
                    indx, indx + (self.tf-self.t0)/self.NumberShootingNodes, self.NrEvaluationPoints),
                    shoots_x_dof[:, indx_shoots], label=label, linestyle=linestyle[dof % len(linestyle)])
                plt.axvline(x=indx, color='k', linestyle='-', linewidth=0.1)
                plt.axvline(x=indx+(self.tf-self.t0)/self.NumberShootingNodes, color='k', linestyle='-', linewidth=0.1)
                indx_shoots += 1
                legend_ok=False
        plt.legend()
        plt.show()
        pass

Ploting = ShootingPlot(t0,tf,NumberShootingNodes,DefineOneShootingNodForDAE_LV, 50);
Ploting.Plot([20., 10.], [0.2, 0.01, 0.001, 0.1], True)

#xdot = x
#ode = {'x': x, 'ode': xdot}

#t0 = 0; tf = 1;
#time = np.insert(np.linspace(t0, tf, 20), t0, 0 )

#intg = integrator('intg', 'cvodes', ode, {'grid': time})

#intg_map = intg.map(2)
#intg_mapaccum = intg.mapaccum(2)

#result = intg_map(x0=[1])
#print(result)
#res_plot = reshape(result['xf'], 20, 2)
#print(res_plot)
##print(res_plot[: ,0])
#plt.plot(np.linspace(0,1,20),res_plot[: ,0])
#plt.plot(np.linspace(1,1+1,20),res_plot[: ,1])
#plt.show()

