# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:00:28 2023

@author: 17000
"""

import torch;
import json;
import numpy as np;
import matplotlib;
from pkgs.dataframe import load_data;
from pkgs.model import V_theta;
from pkgs.deploy import estimator;
import matplotlib.pyplot as plt;
from pkgs.sample_minibatch import sampler;
from pkgs.tomat import to_mat;


device = 'cpu';

molecule_list = ['OH'];

data, labels = load_data(molecule_list, device, ind_list=range(0,20));
sampler1 = sampler(data, labels, device);

est = estimator(device);
est.load('model.pt')

batch_size = 500;
for i in range(len(molecule_list)):
    
    minibatch, labels = sampler1.sample(batch_size=batch_size, i_molecule=i);
    
    Ehat, E = est.solve(minibatch,labels,
                        save_filename=molecule_list[i]);        

est.plot(molecule_list)

