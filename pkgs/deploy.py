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


class estimator():

    def __init__(self, device, ngrid=30) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.model = V_theta(device).to(device)
        self.ngrid = ngrid
        self.integrator = integrate(device)
        self.loss = torch.nn.MSELoss();
    
    def load(self, filename):
        
        self.model.load_state_dict(torch.load(filename));
    
    def solve(self, data_in) -> float:

        # Train the model using given data points
        # M: number of data points, N: number of atoms, B: number of basis
        # pos_list: MxNx3 list of coordinates of input configurations
        # elements_list: MxN list of atomic species, 'C' or 'H'
        # E_list: M list of energy, unit Hartree
        # S_list: MxBxB list of overlap matrix <phi_i|phi_j>
        # N_list: MxBxB list of density matrix <phi_i|N|phi_j>
        # steps: steps to train using this dataset.
        # This method implements gradient descend to the contained model
        # and return the average loss

        S = self.integrator.calc_S_deploy(data_in);
        LB, PhiB = torch.linalg.eigh(S);
        PhiB = torch.einsum('ijk,ik->ijk',[PhiB,LB**(-1/2)]);

        # number of occupied orbitals
        ne = int(sum([1+5*(ele=='C') for ele in data_in['elements']])/2);
        
        V = self.model(data_in);

        A =torch.matmul(torch.matmul(PhiB.permute(0,2,1), V.detach()), PhiB);
        LA, PhiA = torch.linalg.eigh(A);
        
        self.epsilon = LA;
        self.Phi = torch.matmul(PhiB,PhiA);
        self.E = 2*torch.sum(LA[:, :ne], axis=1);
    
        return {'energy': self.E, 'orbital_energy':self.epsilon, 
                'eigenfunction': self.Phi};
    
    
    
    
    