# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:42:41 2023

@author: 17000
"""

from pkgs.integral import integrate
from pkgs.model import V_theta
import numpy as np
import torch
from pkgs.sample_minibatch import sampler;
from pkgs.tomat import to_mat;
from torch.nn.parallel import DistributedDataParallel as DDP;

class trainer():

    def __init__(self, device, data_in, labels, filename='model.pt', lr=10**-3) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.lr = lr
        self.model = V_theta(device).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.integrator = integrate(device)
        self.loss = torch.nn.MSELoss();
        self.filename = filename;
        self.sampler = sampler(data_in, labels, device);
        self.n_molecules = len(data_in);
        self.transformer = to_mat(device);
        
    def find_min_values(self,a, b, N):
        """
        Find the minimum N values in two sorted lists a and b.
        Returns the number of values from a and b respectively.
        """
        i, j = 0, 0
        count_a, count_b = 0, 0
        total_count = 0

        # Iterate until N values are found
        while total_count < N:
            # Check if we've reached the end of either list
            if i >= len(a):
                j += 1
                count_b += 1
            elif j >= len(b):
                i += 1
                count_a += 1
            # Compare elements from both lists and pick the smaller one
            elif a[i] < b[j]:
                i += 1
                count_a += 1
            else:
                j += 1
                count_b += 1
            
            total_count += 1

        return count_a, count_b
        
    def train(self, kE = 0.9, steps=10, batch_size = 50) -> float:

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
        
        L_ave = np.zeros(2);
        
        for _ in range(steps):  # outer loop of training steps
        
            ########### forward calculations ################
            # apply the NN-model to get K-S potential
            self.optim.zero_grad()  # clear gradient
            
            for i in range(self.n_molecules):
                
                minibatch, labels = self.sampler.sample(batch_size=batch_size, i_molecule=i);
                
                h0 = labels['h0'];
                h1 = labels['h1'];
                
                # number of occupied orbitals
                ne = labels['ne'];
                
                E = labels['Ee'];  # ccsdt total energy
                
                V_raw0, V_raw1 = self.model(minibatch);
                
                V0 = self.transformer.raw_to_mat(V_raw0,minibatch,labels);
                V1 = self.transformer.raw_to_mat(V_raw1,minibatch,labels);
                
                H0 = h0 + V0;
                H1 = h1 + V1;
    
                LA0, Phi0 = torch.linalg.eigh(H0.detach());
                LA1, Phi1 = torch.linalg.eigh(H1.detach());
                
                # find out ne0 ad ne1 for each frame
                ne0_l = []
                ne1_l = []
                Ehat = 0
                LE = [0]*batch_size
                for i in range(batch_size):
                    ne0, ne1 = self.find_min_values(LA0[i],LA1[i],ne)
                    Ehat += torch.sum(LA0[:, :ne0], axis=1)+torch.sum(LA1[:, :ne1], axis=1);
                    co0_i = Phi0[i, :, :ne0];
                    co1_i = Phi1[i, :, :ne1];
                    LE[i] = (Ehat-E)*((2*torch.einsum('ij,kj,ik->',
                                                [co0_i, co0_i, H0[i]]))+
                                (2*torch.einsum('ij,kj,ik->',
                                                [co1_i, co1_i, H1[i]])));  # energy term loss function for gradient
                #print('E =',E,'Ehat =',Ehat);
                LE = torch.cat(LE);
                LV = (torch.mean(V0**2)+torch.mean(V1**2))/2;
                L = (kE*torch.mean(LE) + (1-kE)*LV)/self.n_molecules;
                L_ave[1] += self.loss(Ehat,E);
                L_ave[0] += LV;
            ########### calculate loss for output #######
                L.backward()  # calculate the gradient

            self.optim.step()  # implement gradient descend
        
        torch.save(self.model.state_dict(), self.filename);
        
        return L_ave/steps/self.n_molecules;
