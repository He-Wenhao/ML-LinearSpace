#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:39:10 2023

@author: ubuntu
"""

import json;
import torch;
import scipy;

def load_data(molecules, device, load_obs_mat = True, ind_list = [], op_names = []):

    data_in = [];
    labels = [];
    obs_mats = [];
    
    for molecule in molecules:
        with open('data/'+molecule+'_data.json', 'r') as file:

            data = json.load(file);

        if(ind_list == []):

            pos = torch.tensor(data['coordinates']).to(device);
            pos[:,:,[0,1,2]] = pos[:,:,[1,2,0]];
            elements = data['elements'][0];

            ne = int(round(sum([1+5*(ele=='C') for ele in elements])/2));
            norbs = int(round(sum([5+9*(ele=='C') for ele in elements])));
            nframe = len(pos);

            data_in.append({'pos':pos,'elements':elements,
                       'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}});

            h = torch.tensor(data['h']).to(device);
            S_mhalf = [scipy.linalg.fractional_matrix_power(si, (-1/2)).tolist() for si in data['S']]
            S_mhalf = torch.tensor(S_mhalf).to(device);
            h = torch.matmul(torch.matmul(S_mhalf, h),S_mhalf);

            label = {'S': torch.tensor(data['S']).to(device),
                      'h': h,
                      'E': torch.tensor(data['energy']).to(device),
                      'E_nn':torch.tensor(data['Enn']).to(device)}
            for op_name in op_names:
                label[op_name] = torch.tensor(data[op_name]).to(device)
            labels.append(label);
            
            if(load_obs_mat):
                with open('data/'+molecule+'_obs_mat.json', 'r') as file:
                    data_obs = json.load(file);
    
                if op_names is None:
                    op_names = set(data_obs.keys())
    
                obs_mats.append({op: torch.matmul(torch.matmul(S_mhalf,torch.Tensor(data_obs[op]).to(device)),S_mhalf)
                                                                for op in op_names})
                
        else:
            pos = torch.tensor(data['coordinates'])[ind_list].to(device);
            pos[:,:,[0,1,2]] = pos[:,:,[1,2,0]];
            elements = data['elements'][0];

            ne = int(round(sum([1+5*(ele=='C') for ele in elements])/2));
            norbs = int(round(sum([5+9*(ele=='C') for ele in elements])));
            nframe = len(pos);

            data_in.append({'pos':pos,'elements':elements,
                       'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}});

            h = torch.tensor(data['h'])[ind_list].to(device);
            S_mhalf = [scipy.linalg.fractional_matrix_power(data['S'][si], (-1/2)).tolist() for si in ind_list]
            S_mhalf = torch.tensor(S_mhalf).to(device);
            h = torch.matmul(torch.matmul(S_mhalf, h),S_mhalf);

            label = {'S': torch.tensor(data['S'])[ind_list].to(device),
                      'h': h,
                      'E': torch.tensor(data['energy'])[ind_list].to(device),
                      'E_nn':torch.tensor(data['Enn'])[ind_list].to(device)}
            for op_name in op_names:
                label[op_name] = torch.tensor(data[op_name])[ind_list].to(device)
            labels.append(label);
            
            if(load_obs_mat):
                with open('data/'+molecule+'_obs_mat.json', 'r') as file:
                    data_obs = json.load(file);                  
    
                if op_names is None:
                    op_names = set(data_obs.keys())
    
                obs_mats.append({op: torch.matmul(torch.matmul(S_mhalf,torch.Tensor(data_obs[op])[ind_list].to(device)),S_mhalf)
                                                                for op in op_names})
                
    return data_in, labels, obs_mats;


