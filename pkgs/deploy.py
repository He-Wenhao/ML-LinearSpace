#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:54:56 2023

@author: ubuntu
"""

from pkgs.integral import integrate
from pkgs.model import V_theta
import numpy as np
import torch
from pkgs.tomat import to_mat;
import os;
import json;
import matplotlib;
import matplotlib.pyplot as plt;

class estimator():

    def __init__(self, device, output_folder='test_output') -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.integrator = integrate(device)
        self.loss = torch.nn.MSELoss();
        self.transformer = to_mat(device);
        self.output_folder = output_folder;
        if(not os.path.exists(output_folder)):
            os.mkdir(output_folder);

    def load(self, filename):

        self.model = V_theta(self.device).to(self.device)

        try:
            self.model.load_state_dict(torch.load(filename));
        except:
            res = torch.load(filename);
            for key in list(res.keys()):
                res[key[7:]] = res[key];
                del res[key];
            self.model.load_state_dict(res);

    def set_op_matrices(self, op_matrices):
        self.op_matrices = op_matrices

    def solve(self, minibatch, labels, op_names=None, save_filename='data') -> float:

        h = labels['h'];

        # number of occupied orbitals
        ne = labels['ne'];

        E = labels['Ee'];  # ccsdt total energy

        V_raw = self.model(minibatch);

        V = self.transformer.raw_to_mat(V_raw,minibatch,labels);

        H = h + V;

        LA, Phi = torch.linalg.eigh(H.detach());
        Ehat = 2*torch.sum(LA[:, :ne], axis=1);

        if op_names is not None:
            ops = {}
            co = Phi[:, :, :ne]
            for op_name in op_names:
                O = labels[op_name]
                O_mat = self.op_matrices[op_name]
                Ohat = 2 * torch.einsum('jui,uv,jvi->j', [co, O_mat, co])
                ops[op_name] = Ohat
            with open(self.output_folder+'/'+save_filename+'_obs.json','w') as file:
                json.dump(ops,file)

        with open(self.output_folder+'/'+save_filename+'.json','w') as file:
            json.dump({'Ee':E.tolist(), 'Ehat':Ehat.tolist()},file)


        return Ehat, E, ops;

    def plot(self, molecule_list):

        font = {'size' : 18}

        matplotlib.rc('font', **font)
        plt.figure(figsize=(6*len(molecule_list),5.5));
        res = [];
        for i in range(len(molecule_list)):

            with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                data = json.load(file);

            E = data['Ee'];
            Ehat = data['Ehat'];

            plt.subplot(1,len(molecule_list),i+1);

            plt.scatter(np.array(E)*27.211,np.array(Ehat)*27.211);
            plt.title(molecule_list[i]);
            plt.xlabel('E$_{CCSDT}$  (eV)');
            plt.ylabel('E$_{NN}$  (eV)');
            res.append(np.mean((np.array(E)-np.array(Ehat))**2));

        plt.tight_layout();

        print('Standard deviation error:');
        print(str(np.sqrt(np.mean(res))/1.594*10**3)+' kcal/mol');





