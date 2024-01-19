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

    def __init__(self, device, output_folder='test_output', scaling = 0.1) -> None:

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
        self.scaling = scaling;

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

    def solve(self, minibatch, labels, obs_mat, E_nn, data_in,
              op_names=None, atomic_charge = [], save_filename='data') -> float:
        
        angstron2Bohr = 1.88973
        h = labels['h'];

        # number of occupied orbitals
        ne = labels['ne'];

        E = labels['Ee'];  # ccsdt total energy

        V_raw = self.model(minibatch);

        V = self.transformer.raw_to_mat(V_raw,minibatch,labels)*self.scaling;

        H = h + V;

        LA, Phi = torch.linalg.eigh(H.detach());
        Ehat = 2*torch.sum(LA[:, :ne], axis=1);
        
        obj = {'E':(E+E_nn).tolist(), 'Ehat':(Ehat+E_nn).tolist()};

        elements = data_in['elements'];
        nuclearCharge = [1+5*(ele=='C') for ele in elements];
        nuclearCharge = torch.tensor(nuclearCharge, dtype=torch.float).to(self.device);
        mass = torch.tensor([1.008 + (12.011-1.008)*(ele=='C') for ele in elements], dtype=torch.float).to(self.device);
        pos = data_in['pos'];
        mass_center = torch.sum(pos*mass[None,:,None], axis=1)/torch.sum(mass);
        pos = (pos - mass_center[:,None,:])*angstron2Bohr;

        if op_names is not None:

            co = Phi[:, :, :ne]
            for op_name in op_names:
                moment = torch.tensor([op_name.count('x'),
                            op_name.count('y'),
                            op_name.count('z')], dtype=torch.float).to(self.device);
                multipole = torch.sum(torch.prod(pos**moment[None,None,:],
                                                 axis=2)*nuclearCharge[None,:], axis=1);
                
                O = labels[op_name]
                O_mat = obs_mat[op_name]
                Ohat = 2 * torch.einsum('jui,juv,jvi->j', [co, O_mat, co])
                O = multipole - O;
                Ohat = multipole - Ohat;
                obj[op_name] = {'Ohat': Ohat.tolist(),
                                'O': O.tolist()};
        
        if(len(atomic_charge)!=0):
            C = labels['atomic_charge'];

            Cmat = atomic_charge;

            Chat = 2 * torch.einsum('jui,jsuv,jvi->js', [co, Cmat, co]);

            Chat = nuclearCharge[None,:]-Chat;
            C  = nuclearCharge[None,:]-C;
            obj['atomic_charge'] = {'Chat': Chat.tolist(),
                                    'C': C.tolist()};

        with open(self.output_folder+'/'+save_filename+'.json','w') as file:
            json.dump(obj,file)

        return Ehat, E;

    def plot(self, molecule_list, nrows=1):

        font = {'size' : 18}
        
        ncols = int((len(molecule_list)-1)//nrows)+1; 
        
        matplotlib.rc('font', **font)
        plt.figure(figsize=(6*ncols,5.5*nrows));
        res = [];
        for i in range(len(molecule_list)):

            with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                data = json.load(file);

            E = data['E'];
            Ehat = data['Ehat'];

            plt.subplot(nrows,ncols,i+1);
            
            plt.plot([np.min(E)*27.211,np.max(E)*27.211], [np.min(E)*27.211,np.max(E)*27.211], 
                     linestyle='dashed', linewidth = 3,c='red');
            plt.scatter(np.array(E)*27.211,np.array(Ehat)*27.211);
            
            plt.title(molecule_list[i]);
            plt.xlabel('E$_{CCSDT}$  (eV)');
            plt.ylabel('E$_{NN}$  (eV)');
            res.append(np.mean((np.array(E)-np.array(Ehat))**2));

        plt.tight_layout();

        print('Standard deviation error:');
        print(str(np.sqrt(np.mean(res))/1.594*10**3)+' kcal/mol');
        plt.savefig(self.output_folder+'/E.png');
        plt.close();
        ops_dic = {'dipole':['x','y','z'],
                   'quadrupole_diag':['xx','yy','zz'],
                   'quadrupole_offdiag':['xy','xz','yz']};
        
        for ops in ops_dic:
            font = {'size' : 18}
            
            ncols = int((len(molecule_list)-1)//nrows)+1; 
            
            matplotlib.rc('font', **font)
            plt.figure(figsize=(6*ncols,5.5*nrows));
            res = [];
            
            if(ops=='dipole'):
                unit = 'Debye'
            else:
                unit = 'e*a$_0^2$';
                
            for i in range(len(molecule_list)):

                with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                    data = json.load(file);

                O, Ohat = [], [];
                for opsi in ops_dic[ops]:
                    O += data[opsi]['O'];
                    Ohat += data[opsi]['Ohat'];
                
                plt.subplot(nrows,ncols,i+1);

                plt.plot([np.min(O),np.max(O)], [np.min(O),np.max(O)], 
                         linestyle='dashed', linewidth = 3,c='red');
                plt.scatter(np.array(O),np.array(Ohat));
                plt.title(molecule_list[i]);
                plt.xlabel(ops+'$_{CCSDT}$  '+unit);
                plt.ylabel(ops+'$_{NN}$  '+unit);
                res.append(np.mean((np.array(O)-np.array(Ohat))**2));

            plt.tight_layout();

            print(ops+': Standard deviation error:');
            print(str(np.sqrt(np.mean(res)))+' '+unit);
            plt.savefig(self.output_folder+'/'+ops+'.png');
            plt.close();
        
        if('atomic_charge' in data):
            font = {'size' : 18}
            
            ncols = int((len(molecule_list)-1)//nrows)+1; 
            
            matplotlib.rc('font', **font)
            plt.figure(figsize=(6*ncols,5.5*nrows));
            res = [];
            
            for i in range(len(molecule_list)):

                with open(self.output_folder+'/'+molecule_list[i]+'.json','r') as file:

                    data = json.load(file);

                C, Chat = [], [];
                for u in range(len(data['atomic_charge']['C'])):
                    C += data['atomic_charge']['C'][u];
                    Chat += data['atomic_charge']['Chat'][u];

                plt.subplot(nrows,ncols,i+1);

                plt.plot([np.min(C),np.max(C)], [np.min(C),np.max(C)], 
                         linestyle='dashed', linewidth = 3,c='red');
                plt.scatter(np.array(C),np.array(Chat));
                plt.title(molecule_list[i]);
                plt.xlabel('atomic charge$_{CCSDT}$');
                plt.ylabel('atomic charge$_{NN}$');
                res.append(np.mean((np.array(C)-np.array(Chat))**2));

            plt.tight_layout();

            print('atomic charge: Standard deviation error:');
            print(str(np.sqrt(np.mean(res)))+' e');
            plt.savefig(self.output_folder+'/atomic_charge.png');
            plt.close();
