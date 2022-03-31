from FeatherstoneDynamics import *
from casadi import *

urdf_path = "finalProj_Input/robot3.urdf"
root_link = "link Root"
end_link = "link C"

#root_link = "base_link"
#end_link = "tool0" #"shoulder_link"

#root_link = "calib_kuka_arm_base_link"
#end_link = "kuka_arm_7_link"

#root_link = "base_link"
#end_link = "shoulder_link"

robot = URDFparser()
robot.from_file(urdf_path)


# joint_list, joint_names, q_max, q_min = robot.get_joint_info(root_link, end_link)
n_joints = robot.get_n_joints(root_link, end_link)
#n_joints = 3
#
# print("Nr of joints: ", n_joints,"\n")
# for indx in range(0,n_joints):
#
#     print("Name of the joint[ ", indx," ]:", joint_names[indx], "\n")
#     print("Joint information for joint[ ",indx," ]:\n", joint_list[indx])
#     print("------\n\n")
#
# print("\n q max:", q_max)
# print("\n q min:", q_min)
# print('------\n------')

gravity = [0, 0, -9.81]

tau = np.zeros(n_joints)
qddot_sym = robot.get_forward_dynamics_crba(root_link, end_link)
print(qddot_sym)

qddot_g_sym = robot.get_forward_dynamics_aba(root_link, end_link, gravity = gravity)
print(qddot_g_sym)

qddot_sym_exp1, q1, q_dot1, tau1 = robot.get_forward_dynamics_crba_exp(root_link, end_link, gravity = gravity)

Jac_exp = jacobian( qddot_sym_exp1, vertcat( q1, q_dot1,tau1 ) )
Jac_F = Function('Jac_F', [q1, q_dot1,tau1],[Jac_exp],['q','q_dot','tau'],['Jac'],
                 {
                     'print_time' : True,               #(debug)
                 })
#print(Jac_F(q=[0,1,1],q_dot=[0,0,1],tau=[0,0,1])['Jac'])

hess_exp = jacobian( Jac_exp, vertcat( q1, q_dot1,tau1 ) )
Hess_F = Function('Hess_F', [q1, q_dot1,tau1],[hess_exp],['q','q_dot','tau'],['Hess'],
                  {
                      'print_time' : True,               #(debug)
                  })
#print(Hess_F(q=[0,1,1],q_dot=[0,0,1],tau=[0,0,1])['Hess'])


I = integrator(
    'I',
    'cvodes', #cvodes/idas/rk/collocation
    {
        'x': vertcat(q1,q_dot1),
        'p':tau1,
        'ode': vertcat(
            q_dot1[0],
            qddot_sym_exp1[0],
            q_dot1[1],
            qddot_sym_exp1[1],
            q_dot1[2],
            qddot_sym_exp1[2]
        )
    },
    {
        't0': 0, 'tf': 1,
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

print(I(x0=[0,1,0,0,0,0],p=[0,0,0])['xf'])

I_accum = I.mapaccum(
    22,
    {
        #'verbose': True,
        'print_time' : True
    }
)

#print("I_accum:", I_accum( x0=[0,1,0,0,0,0], p=[1,1,0])['xf'] )




q = [None]*n_joints
q_dot = [None]*n_joints
for i in range(n_joints):
    #to make sure the inputs are within the robot's limits:
    #q[i] = (q_max[i] - q_min[i])*np.random.rand()-(q_max[i] - q_min[i])/2
    q[i] = 3
    #q_dot[i] = (q_max[i] - q_min[i])*np.random.rand()-(q_max[i] - q_min[i])/2
    q_dot[i]= 0.1


#qddot_num = qddot_sym(q, q_dot, tau)
qddot_g_num = qddot_g_sym(q, q_dot, tau)
tau = [1,0,0]

#print("Numerical forward dynamics: \n", qddot_num)
#print("\nNumerical forward dynamics w/ gravity: \n", qddot_g_num)


M_sym = robot.get_inertia_matrix_crba(root_link, end_link)
C_sym = robot.get_coriolis_rnea(root_link, end_link)
G_sym = robot.get_gravity_rnea(root_link, end_link, gravity)

M_num = M_sym(q)
C_num = C_sym(q, q_dot)
G_num = G_sym(q)

#M(q) * q_ddot + d_dot^t * C * d_dot = Tau + G(q)
#q_ddot = M(q)^{-1} * ( Tau + G(q) - d_dot^t * C * d_dot )

#cs.solve(M_num, cs.SX.eye(M_num.size1()))
#print("M:", M_num)
#print("M_{-1}:", cs.solve(M_num, cs.SX.eye(M_num.size1())))



##M_inv = cs.solve(M_num, cs.SX.eye(M_num.size1()))
##q_ddot = M_inv * ( tau + G_sym - )


#print("Numerical Inertia Matrx for random input: \n", M_num)
#print("\nNumerical Coriolis term for random input: \n", C_num)
#print("\nNumerical gravity term for random input: \n", G_num)



