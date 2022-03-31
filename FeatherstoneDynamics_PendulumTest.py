from FeatherstoneDynamics import *
from casadi import *
import matplotlib.pyplot as plt

import numpy as np

urdf_path = "finalProj_Input/pendulum.urdf"
root_link = "link Root"
end_link = "link A"

robot = URDFparser()
robot.from_file(urdf_path)

n_joints = robot.get_n_joints(root_link, end_link)

gravity = [0, 0, -9.81]

qddot_sym_exp1, q1, q_dot1, tau1 = robot.get_forward_dynamics_crba_exp(root_link, end_link, gravity = gravity)

I = integrator(
    'I',
    'cvodes', #cvodes/idas/rk/collocation
    {
        'x': vertcat(q1,q_dot1),
        'p':tau1,
        'ode': vertcat(
            q_dot1[0], qddot_sym_exp1[0]
        )
    },
    {
        't0': 0, 'tf': 0.1,
        #'grid': [ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ],
        #'enable_fd' : True,
        #'enable_forward' : False, 'enable_reverse' : False, 'enable_jacobian' : False,
        #'fd_method' : 'smoothing',
        #'print_in':True,                   #print input value parameters(debug)
        #'print_out':True,                  #print output value parameters(debug)
        #'show_eval_warnings':True,         #(debug)
        #'inputs_check':True,               #(debug)
        #'regularity_check':True,           #(debug)
        #'verbose':True,                    #(debug)
        #'print_time' : True,               #(debug)
        #'simplify':True, 'expand':True,
        #'number_of_finite_elements': 5     # rk
        #'max_order' : 4                    #idas/cvodes 2/3/4/5
        #'max_multistep_order': 5           #idas/cvodes 2..5
        #'dump_in' : True,                  #(debug)
        #'dump_out' : True,                 #(debug)
    }
)
print("I (intg):", I )

print(I(x0=[0,0],p=[1])['xf'])

I_accum = I.mapaccum(
    100,
    {
        #'verbose': True,
        'print_time' : True
    }
)

t = np.linspace(0,10,100)
I_accum_res = I_accum( x0=[0,0], p=[1])['xf']
print("I_accum:", I_accum( x0=[0,0], p=[1])['xf'] )

plt.plot(t,I_accum_res[:,:].T)
plt.show()