# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:00:28 2023

@author: 17000
"""

from pkgs.dataframe import load_data;
from pkgs.deploy import estimator;
from pkgs.sample_minibatch import sampler;


device = 'cuda:0';

molecule_list = ['methane','ethane','ethylene','acetylene',
                 'propane','propylene','propyne','cyclopropane'];

OPS = {
       'x':0.1, 'y':0.1, 'z':0.1,
       'xx':0.1, 'yy':0.1, 'zz':0.1,
       'xy':0.1, 'yz':0.1, 'xz':0.1};

data, labels, obs_mat = load_data(molecule_list, device, 
                                  ind_list=range(200),op_names=list(OPS.keys())
                                  );

sampler1 = sampler(data, labels, device);

est = estimator(device);
est.load('model.pt');

batch_size = 200;

for i in range(len(molecule_list)):
    E_nn = labels[i]['E_nn']
    minibatch, labels1 = sampler1.sample(batch_size=batch_size, i_molecule=i,
                                        op_names=list(OPS.keys()));
    
    Ehat, E = est.solve(minibatch, labels1, obs_mat[i], E_nn,
                        save_filename=molecule_list[i],
                        op_names = list(OPS.keys()));

est.plot(molecule_list, nrows=2)

