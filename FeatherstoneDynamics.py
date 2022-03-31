from urdf_parser_py.urdf import URDF, Pose
import numpy as np
from casadi import *

def prismatic(xyz, rpy, axis, qi):
    T = SX.zeros(4, 4)
    cr = cos(rpy[0])
    sr = sin(rpy[0])
    cp = cos(rpy[1])
    sp = sin(rpy[1])
    cy = cos(rpy[2])
    sy = sin(rpy[2])
    r00 = cy*cp
    r01 = cy*sp*sr - sy*cr
    r02 = cy*sp*cr + sy*sr
    r10 = sy*cp
    r11 = sy*sp*sr + cy*cr
    r12 = sy*sp*cr - cy*sr
    r20 = -sp
    r21 = cp*sr
    r22 = cp*cr
    p0 = r00*axis[0]*qi + r01*axis[1]*qi + r02*axis[2]*qi
    p1 = r10*axis[0]*qi + r11*axis[1]*qi + r12*axis[2]*qi
    p2 = r20*axis[0]*qi + r21*axis[1]*qi + r22*axis[2]*qi

    T[0, 0] = r00
    T[0, 1] = r01
    T[0, 2] = r02
    T[1, 0] = r10
    T[1, 1] = r11
    T[1, 2] = r12
    T[2, 0] = r20
    T[2, 1] = r21
    T[2, 2] = r22
    T[0, 3] = xyz[0] + p0
    T[1, 3] = xyz[1] + p1
    T[2, 3] = xyz[2] + p2
    T[3, 3] = 1.0
    return T

def revolute(xyz, rpy, axis, qi):
    T = SX.zeros(4, 4)
    cr = cos(rpy[0])
    sr = sin(rpy[0])
    cp = cos(rpy[1])
    sp = sin(rpy[1])
    cy = cos(rpy[2])
    sy = sin(rpy[2])
    r00 = cy*cp
    r01 = cy*sp*sr - sy*cr
    r02 = cy*sp*cr + sy*sr
    r10 = sy*cp
    r11 = sy*sp*sr + cy*cr
    r12 = sy*sp*cr - cy*sr
    r20 = -sp
    r21 = cp*sr
    r22 = cp*cr
    cqi = cos(qi)
    sqi = sin(qi)
    s00 = (1 - cqi)*axis[0]*axis[0] + cqi
    s11 = (1 - cqi)*axis[1]*axis[1] + cqi
    s22 = (1 - cqi)*axis[2]*axis[2] + cqi
    s01 = (1 - cqi)*axis[0]*axis[1] - axis[2]*sqi
    s10 = (1 - cqi)*axis[0]*axis[1] + axis[2]*sqi
    s12 = (1 - cqi)*axis[1]*axis[2] - axis[0]*sqi
    s21 = (1 - cqi)*axis[1]*axis[2] + axis[0]*sqi
    s20 = (1 - cqi)*axis[0]*axis[2] - axis[1]*sqi
    s02 = (1 - cqi)*axis[0]*axis[2] + axis[1]*sqi
    T[0, 0] = r00*s00 + r01*s10 + r02*s20
    T[1, 0] = r10*s00 + r11*s10 + r12*s20
    T[2, 0] = r20*s00 + r21*s10 + r22*s20
    T[0, 1] = r00*s01 + r01*s11 + r02*s21
    T[1, 1] = r10*s01 + r11*s11 + r12*s21
    T[2, 1] = r20*s01 + r21*s11 + r22*s21
    T[0, 2] = r00*s02 + r01*s12 + r02*s22
    T[1, 2] = r10*s02 + r11*s12 + r12*s22
    T[2, 2] = r20*s02 + r21*s12 + r22*s22
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    T[3, 3] = 1.0
    return T

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def inertia_matrix(I):
    return np.array([I[0], I[1], I[2]],
                    [I[1], I[3], I[4]],
                    [I[2], I[4], I[5]])

def MotionCrossProduct(v):
    cross_ = SX.zeros(6, 6)
    cross_[0, 1] = -v[2]
    cross_[0, 2] = v[1]
    cross_[1, 0] = v[2]
    cross_[1, 2] = -v[0]
    cross_[2, 0] = -v[1]
    cross_[2, 1] = v[0]
    cross_[3, 4] = -v[2]
    cross_[3, 5] = v[1]
    cross_[4, 3] = v[2]
    cross_[4, 5] = -v[0]
    cross_[5, 3] = -v[1]
    cross_[5, 4] = v[0]
    cross_[3, 1] = -v[5]
    cross_[3, 2] = v[4]
    cross_[4, 0] = v[5]
    cross_[4, 2] = -v[3]
    cross_[5, 0] = -v[4]
    cross_[5, 1] = v[3]
    return cross_

def ForceCrossProduct(v):
    return -MotionCrossProduct(v).T

def spatial_inertia_matrix_I_c(ixx, ixy, ixz, iyy, iyz, izz, mass):

    I_c = np.zeros([6, 6])
    I_c[:3, :3] = np.array([[ixx, ixy, ixz],
                           [ixy, iyy, iyz],
                           [ixz, iyz, izz]])

    I_c[3, 3] = mass
    I_c[4, 4] = mass
    I_c[5, 5] = mass

    return I_c

def spatial_inertia_matrix_I_O(ixx, ixy, ixz, iyy, iyz, izz, mass, c):

    I_O = np.zeros([6, 6])
    cx = skew_symmetric(c)
    inertia_matrix = np.array([[ixx, ixy, ixz],
                               [ixy, iyy, iyz],
                               [ixz, iyz, izz]])

    I_O[:3, :3] = inertia_matrix + mass * (np.dot(cx, np.transpose(cx)))
    I_O[:3, 3:] = mass * cx
    I_O[3:, :3] = mass * np.transpose(cx)

    I_O[3, 3] = mass
    I_O[4, 4] = mass
    I_O[5, 5] = mass

    return I_O

def rotation_rpy(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([[cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
                     [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
                     [  -sp,             cp*sr,             cp*cr]])

def SpatialForce_Tr_From_Rot_and_Trans(R, r):
    X = SX.zeros(6, 6)
    X[:3, :3] = R.T
    X[3:, 3:] = R.T
    X[:3, 3:] = skew(r) @ R.T
    return X

def Spatial_Tr_From_Rot_and_Trans(R, r):
    X = SX.zeros(6, 6)
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3:, :3] = -R @ skew(r)
    return X

def Spatial_Tr_Revolute(xyz, rpy, axis, qi):
    T = revolute(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    return Spatial_Tr_From_Rot_and_Trans(rotation_matrix.T, displacement)

def Spatial_Tr_Prismatic(xyz, rpy, axis, qi):
    T = prismatic(xyz, rpy, axis, qi)
    rotation_matrix = T[:3, :3]
    displacement = T[:3, 3]
    return Spatial_Tr_From_Rot_and_Trans(rotation_matrix.T, displacement)

def Spatial_Tr_Mat(xyz, rpy):
    rotation_matrix = rotation_rpy(rpy[0], rpy[1], rpy[2])
    return Spatial_Tr_From_Rot_and_Trans(rotation_matrix.T, xyz)

class Robot(object):

    def __init__(self):
        self.robot = None
        self.actuated_types = ["prismatic", "revolute", "continuous"]

    def Load(self, filename):
        self.robot = URDF.from_xml_file(filename)

    def getJointsNumber(self, root, tip):

        assert self.robot,"load a model first!"

        chain = self.robot.get_chain(root, tip)
        nr_joints = 0

        for elem in chain:
            if elem in self.robot.joint_map:
                joint = self.robot.joint_map[ elem ]
                if joint.type in self.actuated_types:
                    nr_joints += 1

        return nr_joints

    def compute_model(self, root, tip, q):

        assert self.robot,"load a model first!"

        Spatial_Inertias = []
        i_Tr_0 = []
        i_Tr_p = []
        S_is = []

        chain = self.robot.get_chain(root, tip)

        prev_joint = None
        n_actuated = 0
        i = 0

        for item in chain:

            if item in self.robot.joint_map:
                joint = self.robot.joint_map[item]

                if joint.type == "fixed":
                    if prev_joint == "fixed":
                        Spatial_Tr_Mat_prev = Spatial_Tr_Mat( joint.origin.xyz, joint.origin.rpy) @ Spatial_Tr_Mat_prev
                    else:
                        Spatial_Tr_Mat_prev = Spatial_Tr_Mat( joint.origin.xyz, joint.origin.rpy )
                    inertia_transform = Spatial_Tr_Mat_prev
                    prev_inertia = spatial_inertia

                elif joint.type in ["revolute", "continuous"]:
                    if n_actuated != 0:
                        Spatial_Inertias.append(spatial_inertia)
                    n_actuated += 1

                    Tr_Revolute = Spatial_Tr_Revolute( joint.origin.xyz, joint.origin.rpy, joint.axis, q[i] )
                    if prev_joint == "fixed":
                        Tr_Revolute = Tr_Revolute @ Spatial_Tr_Mat_prev
                    S_i = SX( [ joint.axis[0], joint.axis[1], joint.axis[2], 0, 0, 0 ] )
                    i_Tr_p.append( Tr_Revolute )
                    S_is.append( S_i )
                    i += 1

                elif joint.type == "prismatic":
                    if n_actuated != 0:
                        Spatial_Inertias.append( spatial_inertia )
                    n_actuated += 1
                    Tr_Revolute = Spatial_Tr_Prismatic( joint.origin.xyz, joint.origin.rpy, joint.axis, q[i] )
                    if prev_joint == "fixed":
                        Tr_Revolute = Tr_Revolute @ Spatial_Tr_Mat_prev
                    S_i = SX( [0, 0, 0, joint.axis[0], joint.axis[1], joint.axis[2] ] )
                    i_Tr_p.append( Tr_Revolute )
                    S_is.append( S_i )
                    i += 1

                prev_joint = joint.type

            if item in self.robot.link_map:
                link = self.robot.link_map[ item ]

                if link.inertial is None:
                    spatial_inertia = np.zeros((6, 6))
                else:
                    I = link.inertial.inertia
                    spatial_inertia = spatial_inertia_matrix_I_O(
                        I.ixx, I.ixy, I.ixz, I.iyy, I.iyz, I.izz,
                        link.inertial.mass, link.inertial.origin.xyz )

                if prev_joint == "fixed":
                    spatial_inertia = prev_inertia + inertia_transform.T @ spatial_inertia @ inertia_transform

                if link.name == tip:
                    Spatial_Inertias.append( spatial_inertia )

        return (i_Tr_p, S_is, Spatial_Inertias)

    def external_forces(self, external_f, f, i_Tr_p):

        assert False, "Not implemented yet!"

        return f

    def Get_Mass(self, I_c, i_Tr_p, S_i, nrJoints, q):

        M = SX.zeros(nrJoints, nrJoints)
        I_c_combinat = [None]*len(I_c)

        for i in range(0, nrJoints):
            I_c_combinat[i] = I_c[i]

        for i in range(nrJoints-1, -1, -1):
            if i != 0:
                I_c_combinat[i-1] = ( I_c[i-1] + i_Tr_p[i].T @ I_c_combinat[i] @ i_Tr_p[i] )

        for i in range(0, nrJoints):
            aux = I_c_combinat[i] @ S_i[i]
            M[i, i] = S_i[i].T @ aux
            j = i
            while j != 0:
                aux = i_Tr_p[j].T @ aux
                j -= 1
                M[i, j] = S_i[j].T @ aux
                M[j, i] = M[i, j]

        return M

    def Coriolis_and_Gravity(self, i_Tr_p, S_i, I_c, q, q_dot, nrJoints, gravity=None, f_ext=None):

        C = SX.zeros(nrJoints)
        v = []
        a = []
        f = []

        for i in range(0, nrJoints):
            v_J = S_i[i] @ q_dot[i]
            if i == 0:
                v.append(v_J)
                if gravity is not None:
                    ag = np.array([0., 0., 0., gravity[0], gravity[1], gravity[2]])
                    a.append( i_Tr_p[i] @ -ag )
                else:
                    a.append(SX([0., 0., 0., 0., 0., 0.]))
            else:
                v.append( i_Tr_p[i] @ v[i-1] + v_J)
                a.append( i_Tr_p[i] @ a[i-1] + MotionCrossProduct(v[i]) @ v_J )

            f.append( I_c[i] @ a[i] + ForceCrossProduct(v[i]) @ I_c[i] @ v[i] )

        #if f_ext is not None:
        #    f = self.external_forces(f_ext, f, i_Tr_0)

        for i in range(nrJoints-1, -1, -1):
            C[i] = S_i[i].T @ f[i]
            if i != 0:
                f[i-1] = f[i-1] + i_Tr_p[i].T @ f[i]

        return C

    def GetForwardDynamics_CRBA_exp(self, root, tip, gravity=None, f_ext=None):

        assert self.robot, "load a model first!"

        nrJoints = self.getJointsNumber(root, tip)
        q = SX.sym("q", nrJoints)
        q_dot = SX.sym("q_dot", nrJoints)
        tau = SX.sym("tau", nrJoints)
        i_Tr_p, S_i, I_c = self.compute_model(root, tip, q)
        M = self.Get_Mass(I_c, i_Tr_p, S_i, nrJoints, q)
        M_inv = solve(M, SX.eye(M.size1()))
        C = self.Coriolis_and_Gravity(i_Tr_p, S_i, I_c, q, q_dot, nrJoints, gravity, f_ext)
        q_ddot = M_inv @ (tau - C)
        return q_ddot, q,q_dot,tau