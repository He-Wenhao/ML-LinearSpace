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
        
    def solve(self, minibatch, labels, save_filename='data') -> float:

        
        h0 = labels['h0'];
        h1 = labels['h1'];
        
        # number of occupied orbitals
        ne = labels['ne'];
        
        E = labels['Ee'];  # ccsdt total energy
        
        nframe = labels['nframe']
        
        V_raw0, V_raw1 = self.model(minibatch);
        
        V0 = self.transformer.raw_to_mat(V_raw0,minibatch,labels);
        V1 = self.transformer.raw_to_mat(V_raw1,minibatch,labels);
        
        H0 = h0 + V0;
        H1 = h1 + V1;

        LA0, Phi0 = torch.linalg.eigh(H0.detach());
        LA1, Phi1 = torch.linalg.eigh(H1.detach());
        
        Ehat = []
        for i in range(nframe):
            ne0, ne1 = self.find_min_values(LA0[i],LA1[i],ne)
            Ehat.append(float(torch.sum(LA0[i, :ne0])+torch.sum(LA1[i, :ne1])));
        
        with open(self.output_folder+'/'+save_filename+'.json','w') as file:
            json.dump({'Ee':E.tolist(), 'Ehat':Ehat},file)
        
        return Ehat, E;
    
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
    
    
    