# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:42:41 2023

@author: 17000
"""

from pkgs.integral import integrate
from pkgs.model import V_theta
import numpy as np
import torch


class trainer():

    def __init__(self, device, filename='model.pt', kn=0.5, lr=10**-3, ngrid=30) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.k_n = kn
        self.lr = lr
        self.model = V_theta(device).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.ngrid = ngrid
        self.integrator = integrate(device)
        self.loss = torch.nn.MSELoss();
        self.filename = filename;
        
    def pretrain(self, data_in, labels, kE=0.1, steps=10) -> float:

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

        L_ave = np.array([0., 0.])  # output average loss

        S = labels['S'];  # overlap matrix
        LB, PhiB = torch.linalg.eigh(S);
        PhiB = torch.einsum('ijk,ik->ijk',[PhiB,LB**(-1/2)]);
        h = labels['h'];
        
        # number of occupied orbitals
        ne = data_in['properties']['ne'];
        
        E = labels['E'];  # ccsdt total energy
        
        for _ in range(steps):  # outer loop of training steps
        
            ########### forward calculations ################
            # apply the NN-model to get K-S potential
            V = self.model(data_in);
            LV = self.loss(V, h);
            if(kE != 0):
                A =torch.matmul(torch.matmul(PhiB.permute(0,2,1), V.detach()), PhiB);
                LA, PhiA = torch.linalg.eigh(A);
                Phi = torch.matmul(PhiB,PhiA);
                Ehat = 2*torch.sum(LA[:, :ne], axis=1);
                co = Phi[:, :, :ne];
                LE = 2*(Ehat-E)*(2*torch.einsum('uij,ukj,uik->u',
                                              [co, co, V]));  # energy term loss function for gradient
            normalizer = sum([torch.linalg.norm(u)**2 for u in self.model.parameters()]);
            if(kE==0):
                L = LV;
                
            else:
                L = LV+kE*torch.mean(LE)+10**-6*normalizer;
                L_ave[1] += self.loss(Ehat,E);
        ########### calculate loss for output #######

            L_ave[0] += LV;
            
            ########### back propagation and optimization #####

            self.optim.zero_grad()  # clear gradient

            L.backward()  # calculate the gradient

            self.optim.step()  # implement gradient descend
        
        torch.save(self.model.state_dict(), self.filename);
        
        return L_ave/steps
    
    def finetune(self, data_in, labels, coef, steps=10):
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

        L_ave = np.array([0., 0., 0.]);  # output average loss
        nframe,nbasis = data_in['properties']['nframe'], data_in['properties']['norbs'];
        S = labels['S'];  # overlap matrix
        LB, PhiB = torch.linalg.eigh(S);

        PhiB = torch.einsum('ijk,ik->ijk',[PhiB,LB**(-1/2)]);
        
        N = labels['N'];  # electron density matrix
        h = labels['h'];

        # number of occupied orbitals
        ne = data_in['properties']['ne'];
        
        for _ in range(steps):  # outer loop of training steps

            Vtheta = self.model(data_in);
            
            LV = self.loss(Vtheta, h);
            A =torch.matmul(torch.matmul(PhiB.permute(0,2,1), Vtheta.detach()), PhiB);
            LA, PhiA = torch.linalg.eigh(A);
            Phi = torch.matmul(PhiB,PhiA);
            ########### forward calculations ################
            
            E_orb = 2*torch.sum(LA[:, :ne], axis=1);

            # predicted total energy
            Ehat = E_orb;
            E = labels['E'];  # ccsdt total energy

            ########### energy part of gradient ##############

            # occupied (co) and vacant (cv) states
            co, cv = Phi[:, :, :ne], Phi[:, :, ne:];
            LE = (Ehat-E)*(2*torch.einsum('uij,ukj,uik->u',
                                          [co, co, Vtheta]));  # energy term loss function for gradient
            LE = torch.mean(torch.mean(LE));
            
            ########### density part of gradient #############
            # calculate c_i * (sum_j ci*F*ci) * c_k
            Fik = self.integrator.calc_F(
                data_in, Phi, ngrid=self.ngrid);
            # calculate c_i * N * c_k
            Nik = torch.einsum('jui,juv,jvk->jik', [co, N, cv]);

            # calculate 1/(epsilon_k-epsilon_i)
            epsilon_ik = (LA[:, :ne].reshape([nframe,ne,1]) -
                          LA[:, ne:].reshape([nframe, 1, nbasis-ne]))**-1;

            # density contribution to the loss for gradient
            Ln = torch.einsum('jik,jik,jmk,jmn,jni->',
                              [Fik-Nik, epsilon_ik, cv, Vtheta, co])*8;

            L = coef['V']*LV + coef['E']*LE + coef['n']*Ln;  # Total loss function for gradient

        ########### calculate loss for output #######

#            L_ave[0] += torch.mean((E-Ehat)**2/2)/steps; # energy term in the loss function
            L_ave[0] += LV;
            L_ave[1] += torch.mean((E-Ehat)**2/2);
            L_ave[2] += torch.mean(torch.norm(Fik-Nik, dim=(1,2))**2,axis=0);
            
            ########### back propagation and optimization #####

            self.optim.zero_grad();  # clear gradient

            L.backward();  # calculate the gradient

            self.optim.step();  # implement gradient descend
        
        torch.save(self.model.state_dict(), self.filename);
        
        return L_ave/steps;