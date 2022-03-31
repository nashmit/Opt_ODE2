from biorbd import *
#from casadi import *
#import biorbd
import numpy as np
#import bioviz

model = biorbd.Model('finalProj_input/pendulum.bioMod')

# Prepare the model
Q = np.ones(model.nbQ())/10  # Set the model position
QDot = np.ones(model.nbQ())/10  # Set the model velocity
states = model.stateSet() # for this example we the size of 'states' is 0 as we only have 1 element
for state in states:
    state.setActivation(0.5)  # Set muscles activations

# Compute the joint torques based on muscle
joint_torque = model.muscularJointTorque(states, Q, QDot)

external_force_vector = np.zeros(6) # Spatial vector has 6 components
f_ext = biorbd.VecBiorbdSpatialVector()
f_ext.append(biorbd.SpatialVector(external_force_vector)) #only one since we have only one object in this
# hierarchy ( e.g. model.nbSegment() ) and each segment has to have one 'SpatialVector' external force. In case
# external forces are applyed to some subset of objects/bodies/Segments then all the other components will still
# need to be provided each one with a '[0]' 'SpatialVector' each. The Spatial vector has the first 3 elements as
# Torque and the last 3 elements as forces in cartezian coordinates, defined in 'base coordinates':
# https://rbdl.github.io/d6/d63/group__dynamics__group.html#ga941bad8e3b7323aaa4a333922fd90480
# this forces are "acting at the center of mass of this particular segment." : https://github.com/pyomeca/bioptim

print('Spatial Vector:',f_ext[0].to_array())
# Compute the acceleration of the model due to these torques
QDDot = model.ForwardDynamics(Q, QDot, joint_torque,f_ext)
# Print the results
print(QDDot.to_array())

external_force_vector = np.array([1,2,3,4,5,6]) #np.ones(6)
#f_ext = biorbd.VecBiorbdSpatialVector()
f_ext[0]=biorbd.SpatialVector(external_force_vector)
print('Spatial Vector:',f_ext[0].to_array())
QDDot = model.ForwardDynamics(Q, QDot, joint_torque,f_ext)
print(QDDot.to_array())

# Do some stuff with you model...
#bioviz.Viz(loaded_model=model).exec()


# bioviz.Viz('pendulum.bioMod',
#            show_meshes=True,
#            show_global_center_of_mass=True, show_segments_center_of_mass=True,
#            show_global_ref_frame=True, show_local_ref_frame=True,
#            show_markers=True,
#            show_muscles=True,
#            show_analyses_panel=True
#            ).exec()