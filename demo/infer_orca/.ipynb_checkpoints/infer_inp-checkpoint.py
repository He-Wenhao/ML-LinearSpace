import sys


sys.path.append('src')

from deploy import infer_func;
import os;
# This script is used to apply a pre-trained multi-task electronic structure model to infer properties of molecules

params = {};

# properties to be inferred.
# V: regularization term, E: energy,  x,y,z: electric dipole, F: force on nuclei,
# xx,yy,zz,xy,yz,xz: electric quadrupole, atomic_charge: atomic charge,
# E_gap: optical gap, bond_order: bond order, alpha: electric static polarizability
# The weights are not used in the inference
params['OPS'] = {'proj':1,'pos':1,'elements':1,'name':1};

params['device'] = 'cuda:0'; # device to run the code for calculations in the inference.
params['batch_size'] = 5;    # batch size for inference
params['scaling'] = {'V':1, 'T': 0.01};         # scaling factors for the neural network correction terms.
                                                # V: Hamiltonian correction, T: screening matrix for polarizability.
                                                # should the same as the scaling factors used in the model training.
params['element_list'] = ['H','C','N','O','F']; # list of elements in the dataset
                                                # should be the same as the element_list used in the model training
params['path'] = os.getcwd() +'/';              # path to the package directory
params['datagroup'] = ['H4_diss'];          # list of data groups for inference
params['output_path']= os.getcwd()+'/output_proj/';  # output path for the inference results
params['model_file'] = 'test.pt';         # model file for inference
params['nodeRDM_flag'] = False

infer_func(params); # infer the properties of molecules
