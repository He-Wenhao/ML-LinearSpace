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

    def __init__(self, device, data_in, labels, op_matrices=[], filename='model.pt', lr=10**-3) -> None:

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

        self.op_matrices = op_matrices

    def train(self, steps=10, batch_size = 50,
                    op_names=[]) -> float:

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
        operators_except_EV = [key for key in list(op_names.keys()) if key!='E' and key!='V'];

        L_ave = np.zeros(len(op_names))
        
        for _ in range(steps):  # outer loop of training steps

            ########### forward calculations ################
            # apply the NN-model to get K-S potential
            self.optim.zero_grad()  # clear gradient

            for i_m in range(self.n_molecules):

                minibatch, labels = self.sampler.sample(batch_size=batch_size, i_molecule=i_m,
                                                        op_names=operators_except_EV);

                h = labels['h'];

                # number of occupied orbitals
                ne = labels['ne'];

                nbasis = labels['norbs']
                nframe = labels['nframe']

                E = labels['Ee'];  # ccsdt total energy

                V_raw = self.model(minibatch);

                V = self.transformer.raw_to_mat(V_raw,minibatch,labels);

                H = h + V;

                LA, Phi = torch.linalg.eigh(H.detach());
                Ehat = 2*torch.sum(LA[:, :ne], axis=1);
                co = Phi[:, :, :ne];
                LE = 2*(Ehat-E)*(2*torch.einsum('uij,ukj,uik->u',
                                              [co, co, H]));  # energy term loss function for gradient
                LV = torch.mean(V**2);
                L = (op_names['E']*torch.mean(LE) + op_names['V']*LV)/self.n_molecules;
                L_ave[1] += self.loss(Ehat,E);
                L_ave[0] += LV;

                ######## OPERATOR TRAINING ####################################
                if len(op_names) > 2:
                    LO = 0
                    # calculate 1/(epsilon_k-epsilon_i)
                    epsilon_ik = (LA[:, :ne].reshape([nframe,ne,1]) -
                          LA[:, ne:].reshape([nframe, 1, nbasis-ne]))**-1;
                    cv = Phi[:, :, ne:]
                    for i, op_name in enumerate(op_names):
                        if(op_name!='E' and op_name!='V'):
                            O = labels[op_name]
                            O_mat = self.op_matrices[i_m][op_name];
                            
                            Ohat = 2 * torch.einsum('jui,juv,jvi->j', [co, O_mat, co])
                            O_term = (Ohat - O)[:,None,None] * \
                                     torch.einsum('juk,juv,jvi->jik', [cv, O_mat, co]) * \
                                     torch.einsum('jui,juv,jvk->jik', [co, V, cv])
                            LO = LO + op_names[op_name] * torch.einsum('jik,jik->',epsilon_ik, O_term) * 2
                            L_ave[i] += torch.mean((Ohat - O)**2/2)

                    L = L + (LO / self.n_molecules)

            ########### calculate loss for output #######
                L.backward()  # calculate the gradient

            self.optim.step()  # implement gradient descend


        torch.save(self.model.state_dict(), self.filename);

        return L_ave/steps/self.n_molecules;


class trainer_ddp():

    def __init__(self, device, data_in, labels, 
                 op_matrices=[],filename='model.pt', lr=10**-3) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.lr = lr
        self.model = DDP(V_theta(device).to(device), device_ids=[device]);
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.integrator = integrate(device)
        self.loss = torch.nn.MSELoss();
        self.filename = filename;
        self.sampler = sampler(data_in, labels, device);
        self.n_molecules = len(data_in);
        self.transformer = to_mat(device);
        self.op_matrices = op_matrices;

    def train(self, kE = 0.9, steps=10, batch_size = 50,
              op_names=[]) -> float:

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
        operators_except_EV = [key for key in list(op_names.keys()) if key!='E' and key!='V'];
        L_ave = np.zeros(len(op_names))

        for _ in range(steps):  # outer loop of training steps

            ########### forward calculations ################
            # apply the NN-model to get K-S potential
            self.optim.zero_grad()  # clear gradient

            for i_m in range(self.n_molecules):

                minibatch, labels = self.sampler.sample(batch_size=batch_size, i_molecule=i_m,
                                                        op_names=operators_except_EV);

                h = labels['h'];

                # number of occupied orbitals
                ne = labels['ne'];

                E = labels['Ee'];  # ccsdt total energy
                nbasis = labels['norbs']
                nframe = labels['nframe']
                
                V_raw = self.model(minibatch);

                V = self.transformer.raw_to_mat(V_raw,minibatch,labels);

                H = h + V;

                LA, Phi = torch.linalg.eigh(H.detach());
                Ehat = 2*torch.sum(LA[:, :ne], axis=1);
                co = Phi[:, :, :ne];
                LE = 2*(Ehat-E)*(2*torch.einsum('uij,ukj,uik->u',
                                              [co, co, H]));  # energy term loss function for gradient
                LV = torch.mean(V**2);
                L = (kE*torch.mean(LE) + (1-kE)*LV)/self.n_molecules;
                L_ave[1] += self.loss(Ehat,E);
                L_ave[0] += LV;
                
                ######## OPERATOR TRAINING ####################################
                if len(op_names) > 2:
                    LO = 0
                    # calculate 1/(epsilon_k-epsilon_i)
                    epsilon_ik = (LA[:, :ne].reshape([nframe,ne,1]) -
                          LA[:, ne:].reshape([nframe, 1, nbasis-ne]))**-1;
                    cv = Phi[:, :, ne:]
                    for i, op_name in enumerate(op_names):
                        if(op_name!='E' and op_name!='V'):
                            O = labels[op_name]
                            O_mat = self.op_matrices[i_m][op_name];
                            
                            Ohat = 2 * torch.einsum('jui,juv,jvi->j', [co, O_mat, co])
                            O_term = (Ohat - O)[:,None,None] * \
                                     torch.einsum('juk,juv,jvi->jik', [cv, O_mat, co]) * \
                                     torch.einsum('jui,juv,jvk->jik', [co, V, cv])
                            LO = LO + op_names[op_name] * torch.einsum('jik,jik->',epsilon_ik, O_term) * 2
                            L_ave[i] += torch.mean((Ohat - O)**2/2)

                    L = L + (LO / self.n_molecules)
            ########### calculate loss for output #######
                L.backward()  # calculate the gradient

            self.optim.step()  # implement gradient descend

        if(self.device==0): 
            torch.save(self.model.state_dict(), self.filename);

        return L_ave/steps/self.n_molecules;

