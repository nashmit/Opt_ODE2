from DAE import *

import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar



from casadi import *
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from copy import deepcopy
from typing import Union



class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        #self.show()
        #self.showMaximized()
        #exit(self.qapp.exec_())
        pass

    def showWin(self):
        self.show()
        #exit(self.qapp.exec_())
        self.qapp.exec_()
        pass

    def showWinFull(self):
        self.showMaximized()
        #exit(self.qapp.exec_())
        self.qapp.exec_()
        pass



class ShootingPlot:
    def __init__(self, t0, tf, NumberShootingNodes, DAE_info, NrEvaluationPoints=40, StatesName=[]):

        self.nrSubPlots=50
        self.currentSubPlot=0

        self.StatesName=StatesName

        fig = plt.figure(figsize=(4.7, 1.7*self.nrSubPlots))
        self.window = ScrollableWindow(fig)
        #plt.subplots_adjust(wspace=0.1,hspace=0.5)
        #plt.subplots_adjust(hspace=0)
        plt.subplots_adjust(bottom=0.2,top=0.9,hspace=0.5)

        self.t0 = t0
        self.tf = tf
        self.NumberShootingNodes = NumberShootingNodes

        self.NrEvaluationPoints = NrEvaluationPoints

        intg, SizeOfx0, SizeOfp, I_plot = \
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

    def Plot(self, x0, p, mapaccum=False, addToPlotName='', mergeStates=False):
        # https://stackoverflow.com/questions/13359951/is-there-a-list-of-line-styles-in-matplotlib
        linestyle = {0: '-', 1: '-.', 2: '--', 3: ':'}
        #linestyle = {0:'-'}
        color = {0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y'}
        #color = {0:'b'}

        #set font size for all elements
        plt.rcParams.update({'font.size': 3})


        if mapaccum == False:
            res = self.eval_intg_exp(x0, p)
        else:
            res = self.eval_intg_exp_mapaccum(x0, p)

        if isinstance(x0, list):
            size = len(x0)
        else:
            size = x0.numel()

        for dof in range(0, size):

            indx_shoots = 0
            legend_ok=True

            if dof==0:
                self.currentSubPlot = (self.currentSubPlot + 1) % self.nrSubPlots
            else:
                if mergeStates==False:
                    self.currentSubPlot = (self.currentSubPlot + 1) % self.nrSubPlots

            #plt.subplot( len(x0), 1, dof+1)
            plt.subplot( self.nrSubPlots, 1, self.currentSubPlot + 1)

            for indx in np.arange(self.t0, self.tf, (self.tf-self.t0)/self.NumberShootingNodes):
                if legend_ok:
                    if len(self.StatesName)>0:
                        label = self.StatesName[dof]
                    else:
                        label= 'state ' + str(dof)
                else:
                    label=''
                res_x_dof = res[dof, :]
                shoots_x_dof = reshape(res_x_dof, self.NrEvaluationPoints, self.NumberShootingNodes)
                plt.plot(np.linspace(
                    indx, indx + (self.tf-self.t0)/self.NumberShootingNodes, self.NrEvaluationPoints),
                    shoots_x_dof[:, indx_shoots], label=label, linewidth=0.5,
                    linestyle=linestyle[dof % len(linestyle)], color= color[indx_shoots %len(color)])
                plt.scatter(indx, shoots_x_dof[0, indx_shoots].full() , linewidth=0.5,
                            s=5, facecolors='none', edgecolors=color[indx_shoots %len(color)] )
                plt.axvline(x=indx, color='k', linestyle='-', linewidth=0.1)
                plt.axvline(x=indx+(self.tf-self.t0)/self.NumberShootingNodes, color='k', linestyle='-', linewidth=0.1)
                indx_shoots += 1
                legend_ok=False

            plt.legend()
            plt.xlabel('time(s)')
            plt.ylabel('state value')
            plt.title(addToPlotName, color='r', size=4)
        #plt.suptitle('Dynamical system', color='g')
        #plt.show()
        pass

    def Show(self):
        #plt.show()
        self.window.showWinFull()
        pass


def DefineOneShootingNodForVanDerPol(t0, tf, NrEvaluationPoints = 40):
    x0 = SX.sym('x0')
    x1 = SX.sym('x1')
    x = vertcat(x0, x1)
    c = SX.sym('c')
    k = SX.sym('k')
    p = vertcat(c, k)
    xdot = vertcat(
        x1,
        c * ( 1 - x0 * x0 ) * x1 - k * x0
    )

    ode = { 'x':x, 'p':p, 'ode':xdot }

    I = integrator('I', 'cvodes', ode, {'t0': t0, 'tf': tf})

    time = np.insert(np.linspace(t0, tf, NrEvaluationPoints), t0, 0)
    I_plot = integrator('I_plot', 'cvodes', ode, {'grid': time})

    # returns [integrator][Nr equations][Nr parameters]
    return I, 2, 2, I_plot

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

    I = integrator('I', 'cvodes', ode, {'t0': t0, 'tf': tf});
    time = np.insert(np.linspace(t0, tf, NrEvaluationPoints), t0, 0)
    I_plot = integrator('I_plot', 'cvodes', ode, {'grid': time});

    # returns [integrator][Nr equations][Nr parameters]
    return I, 2, 4, I_plot;

def Pyridine(t0, tf, NrEvaluationPoints = 40):
    A = SX.sym('A')
    B = SX.sym('B')
    C = SX.sym('C')
    D = SX.sym('D')
    E = SX.sym('E')
    F = SX.sym('F')
    G = SX.sym('G')
    x = vertcat(A, B, C, D, E, F, G)
    # p = SX.sym('p',11)
    p1 = SX.sym('p1');
    p2 = SX.sym('p2');
    p3 = SX.sym('p3');
    p4 = SX.sym('p4');
    p5 = SX.sym('p5');
    p6 = SX.sym('p6');
    p7 = SX.sym('p7');
    p8 = SX.sym('p8');
    p9 = SX.sym('p9');
    p10 = SX.sym('p10');
    p11 = SX.sym('p11');
    p = vertcat(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
    xdot = vertcat(-p1 * A + p9 * B,
                   p1 * A - p2 * B - p3 * B * C - p7 * D - p9 * B + p10 * D * F,
                   p2 * B - p3 * B * C - 2 * p4 * C * C - p6 * C + p8 * E + p10 * D * F + 2 * p11 * E * F,
                   p3 * B * C - p5 * D - p7 * D - p10 * D * F,
                   p4 * C * C + p5 * D - p8 * E - p11 * E * F,
                   p3 * B * C + p4 * C * C + p6 * C - p10 * D * F - p11 * E * F,
                   p6 * C + p7 * D + p8 * F)
    ode = {'x': x, 'p': p, 'ode': xdot}

    I = integrator('I', 'cvodes', ode, {'t0': t0, 'tf': tf})

    time = np.insert(np.linspace(t0, tf, NrEvaluationPoints), t0, 0)
    I_plot = integrator('I_plot', 'cvodes', ode, {'grid': time})

    return I, 7, 11, I_plot


#t0 = 0
#tf = 10
#NumberShootingNodes = 20
#Ploting = ShootingPlot(t0,tf,NumberShootingNodes,DefineOneShootingNodForVanDerPol, 50);
#Ploting.Plot(Config_VanDerPol['Sn'],Config_VanDerPol['w']+Config_VanDerPol['q'], True, '(Iteration: 1)', False)
#Ploting.Plot([2., 1.], [1.0, 1.0], True, '(Iteration: 2)', True)
#Ploting.Plot([2., 1.], [1.0, 1.0], True, '(Iteration: 3)')
#Ploting.Plot([2., 1.], [1.0, 1.0], True, '(Iteration: 4)', True)
#Ploting.Plot([2., 1.], [1.0, 1.0], True)
#Ploting.Show()


#t0 = 0
#tf = 100
#NumberShootingNodes = 20
#Ploting = ShootingPlot(t0,tf,NumberShootingNodes,DefineOneShootingNodForDAE_LV, 50);
#Ploting.Plot([20., 10.], [0.2, 0.01, 0.001, 0.1], True, '(Iteration: 1)', True)
#Ploting.Show()


#t0 = 0
#tf = 5.5
#NumberShootingNodes = 20
#GroundTruthPyridine= {
#    'Sn': DM([1, 0, 0, 0, 0, 0, 0]),
#    'p': DM([1.81, 0.894, 29.4, 9.21, 0.0580, 2.43, 0.0644, 5.55, 0.0201, 0.577, 2.15])/4.5,
#    'SnName':['Pyridin', 'Piperidin', 'Pentylamin', 'N-Pentylpiperidin', 'Dipentylamin', 'Ammonia', 'Pentan']
#}
#Ploting = ShootingPlot(t0,tf,NumberShootingNodes,Pyridine, 50, GroundTruthPyridine['SnName'] );
#Ploting.Plot(GroundTruthPyridine['Sn'], GroundTruthPyridine['p'], True, '(Iteration: 1)', True)
#Ploting.Plot(GroundTruthPyridine['Sn'], GroundTruthPyridine['p'], True, '(Iteration: 2)', False)
#Ploting.Show()


class OCP_Plot:
    def __init__(self, t0, tf, NumberShootingNodes, DAE_info,
                 BuildCasADiExpFromIntegrator, NrEvaluationPoints=40, StatesName=[]):

        self.nrSubPlots=100#50
        self.currentSubPlot=0

        self.StatesName=StatesName

        fig = plt.figure(figsize=(4.7, 1.7*self.nrSubPlots))
        self.window = ScrollableWindow(fig)
        #plt.subplots_adjust(wspace=0.1,hspace=0.5)
        #plt.subplots_adjust(hspace=0)
        plt.subplots_adjust(bottom=0.2,top=0.9,hspace=0.5)

        self.t0 = t0
        self.tf = tf
        self.NumberShootingNodes = NumberShootingNodes

        self.NrEvaluationPoints = NrEvaluationPoints

        #intg, SizeOfx0, SizeOfp, I_plot = \
        #    DAE_info(t0, (tf-t0)/NumberShootingNodes, NrEvaluationPoints=NrEvaluationPoints)
        I_plot=DAE_info.GetI_plot_TimeNormalized( t0, tf , NumberShootingNodes, NrEvaluationPoints=NrEvaluationPoints )
        SizeOfx0 = DAE_info.GetX().size1()
        SizeOfp = DAE_info.GetP().size1()

        self.Sn, self.p, self.intg_exp = \
            BuildCasADiExpFromIntegrator(NumberShootingNodes, I_plot, SizeOfx0, SizeOfp, False)
        self.eval_intg_exp = Function('eval_intg_exp', [self.Sn, self.p], [self.intg_exp['xf']], ['x0', 'p'], ['out'])

        self.Sn_mapaccum, self.p_mapaccum, self.intg_exp_mapaccum = \
            BuildCasADiExpFromIntegrator(NumberShootingNodes, I_plot, SizeOfx0, SizeOfp, True)
        self.eval_intg_exp_mapaccum = \
            Function('eval_intg_exp_mapaccum',
                     [self.Sn_mapaccum, self.p_mapaccum], [self.intg_exp_mapaccum['xf']], ['x0', 'p'], ['out'])
    pass

    def Plot(self, x0, p, mapaccum=False, addToPlotName='', mergeStates=False):
        # https://stackoverflow.com/questions/13359951/is-there-a-list-of-line-styles-in-matplotlib
        linestyle = {0: '-', 1: '-.', 2: '--', 3: ':'}
        #linestyle = {0:'-'}
        color = {0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y'}
        #color = {0:'b'}

        #set font size for all elements
        plt.rcParams.update({'font.size': 3})


        if mapaccum == False:
            res = self.eval_intg_exp(x0, p)
        else:
            res = self.eval_intg_exp_mapaccum(x0, p)

        if isinstance(x0, list):
            size = len(x0[0])
        else:
            size = x0[:,0].numel()

        for dof in range(0, size):

            indx_shoots = 0
            legend_ok=True

            if dof==0:
                self.currentSubPlot = (self.currentSubPlot + 1) % self.nrSubPlots
            else:
                if mergeStates==False:
                    self.currentSubPlot = (self.currentSubPlot + 1) % self.nrSubPlots

            #plt.subplot( len(x0), 1, dof+1)
            plt.subplot( self.nrSubPlots, 1, self.currentSubPlot + 1)

            for indx in np.arange(self.t0, self.tf, (self.tf-self.t0)/self.NumberShootingNodes):
                if legend_ok:
                    if len(self.StatesName)>0:
                        label = self.StatesName[dof]
                    else:
                        label= 'state ' + str(dof)
                else:
                    label=''
                res_x_dof = res[dof, :]
                shoots_x_dof = reshape(res_x_dof, self.NrEvaluationPoints, self.NumberShootingNodes)
                plt.plot(np.linspace(
                    indx, indx + (self.tf-self.t0)/self.NumberShootingNodes, self.NrEvaluationPoints),
                    shoots_x_dof[:, indx_shoots], label=label, linewidth=0.5,
                    linestyle=linestyle[dof % len(linestyle)], color= color[indx_shoots %len(color)])
                plt.scatter(indx, shoots_x_dof[0, indx_shoots].full() , linewidth=0.5,
                            s=5, facecolors='none', edgecolors=color[indx_shoots %len(color)] )
                plt.axvline(x=indx, color='k', linestyle='-', linewidth=0.1)
                plt.axvline(x=indx+(self.tf-self.t0)/self.NumberShootingNodes, color='k', linestyle='-', linewidth=0.1)
                indx_shoots += 1
                legend_ok=False

            plt.legend()
            plt.xlabel('time(s)')
            plt.ylabel('state value')
            plt.title(addToPlotName, color='r', size=4)
        #plt.suptitle('Dynamical system', color='g')
        #plt.show()
        pass

    def Show(self):
        #plt.show()
        self.window.showWinFull()
        pass
