#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:39:10 2023

@author: ubuntu
"""

import json;
import torch;

def load_data(molecule, device, ind_list = []):
    
    with open('data/'+molecule+'.json', 'r') as file:
        
        data = json.load(file);
        
    if(ind_list == []):
    
        pos = torch.tensor(data['coordinates']).to(device);
        pos[:,:,[0,1,2]] = pos[:,:,[1,2,0]];
        elements = data['elements'][0];
        
        ne = int(round(sum([1+5*(ele=='C') for ele in elements])/2));
        norbs = int(round(sum([5+9*(ele=='C') for ele in elements])));
        nframe = len(pos);
        
        data_in = {'pos':pos,'elements':elements,
                   'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}};
        
        labels = {'S': torch.tensor(data['S']).to(device),
                  'N': torch.tensor(data['N']).to(device),
                  'h': torch.tensor(data['h']).to(device),
                  'E': torch.tensor(data['energy']).to(device),
                  'E_nn':torch.tensor(data['Enn']).to(device)};
    else:
        pos = torch.tensor(data['coordinates'])[ind_list].to(device);
        pos[:,:,[0,1,2]] = pos[:,:,[1,2,0]];
        elements = data['elements'][0];
        
        ne = int(round(sum([1+5*(ele=='C') for ele in elements])/2));
        norbs = int(round(sum([5+9*(ele=='C') for ele in elements])));
        nframe = len(pos);
        
        data_in = {'pos':pos,'elements':elements,
                   'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}};
        
        labels = {'S': torch.tensor(data['S'])[ind_list].to(device),
                  'N': torch.tensor(data['N'])[ind_list].to(device),
                  'h': torch.tensor(data['h'])[ind_list].to(device),
                  'E': torch.tensor(data['energy'])[ind_list].to(device),
                  'E_nn':torch.tensor(data['Enn'])[ind_list].to(device)};
        
    return data_in, labels;  