# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:00:28 2023

@author: 17000
"""

import torch;
import json;
import numpy as np;
from pkgs.train import trainer;
from pkgs.dataframe import load_data;
from pkgs.model import V_theta;
from pkgs.deploy import estimator;

device = 'cpu';
est = estimator(device);
data, labels = load_data('Methane', device,
                         ind_list = [0]);

est.load('model.pt');

results = est.solve(data);
