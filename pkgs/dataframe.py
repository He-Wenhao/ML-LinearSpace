#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:39:10 2023

@author: ubuntu
"""

import json;
import torch;
import scipy;
from scipy.linalg import sqrtm, inv

def load_data(molecules, device, ind_list = []):
    
    data_in = [];
    labels = [];
    
    for molecule in molecules:
        with open('data/'+molecule+'_data.json', 'r') as file:
            
            data = json.load(file);
            
        if(ind_list == []):
        
            pos = torch.tensor(data['coordinates']).to(device);
            pos[:,:,[0,1,2]] = pos[:,:,[1,2,0]];
            elements = data['elements'][0];
            
            ne = int(round(sum([1+7*(ele=='O') for ele in elements])));
            norbs = int(round(sum([5+9*(ele=='O') for ele in elements])));
            nframe = len(pos);
            
            data_in.append({'pos':pos,'elements':elements,
                       'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}});
            
            h0 = torch.tensor(data['h0']).to(device);
            h1 = torch.tensor(data['h1']).to(device);
            S_mhalf = [scipy.linalg.fractional_matrix_power(si, (-1/2)).tolist() for si in data['S']]
            S_mhalf = torch.tensor(S_mhalf).to(device);
            S_mhalf = S_mhalf.real;
            h0 = torch.matmul(torch.matmul(S_mhalf, h0),S_mhalf);
            h1 = torch.matmul(torch.matmul(S_mhalf, h1),S_mhalf);
            
            labels.append({'S': torch.tensor(data['S']).to(device),
                      'h0': h0,
                      'h1': h1,
                      'E': torch.tensor(data['energy']).to(device),
                      'E_nn':torch.tensor(data['Enn']).to(device)});
        else:
            pos = torch.tensor(data['coordinates'])[ind_list].to(device);
            pos[:,:,[0,1,2]] = pos[:,:,[1,2,0]];
            elements = data['elements'][0];
            
            ne = int(round(sum([1+7*(ele=='O') for ele in elements])));
            norbs = int(round(sum([5+9*(ele=='O') for ele in elements])));
            nframe = len(pos);
            
            data_in.append({'pos':pos,'elements':elements,
                       'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}});
            
            h0 = torch.tensor(data['h0'])[ind_list].to(device);
            h1 = torch.tensor(data['h1'])[ind_list].to(device);
            S_mhalf = [scipy.linalg.fractional_matrix_power(data['S'][si], (-1/2)).tolist() for si in ind_list]
            S_mhalf = torch.tensor(S_mhalf).to(device);
            S_mhalf = S_mhalf.real;
            h0 = torch.matmul(torch.matmul(S_mhalf, h0),S_mhalf);
            h1 = torch.matmul(torch.matmul(S_mhalf, h1),S_mhalf);
            
            labels.append({'S': torch.tensor(data['S'])[ind_list].to(device),
                      'h0': h0,
                      'h1': h1,
                      'E': torch.tensor(data['energy'])[ind_list].to(device),
                      'E_nn':torch.tensor(data['Enn'])[ind_list].to(device)});
            
    return data_in, labels;  