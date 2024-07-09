import torch;
import json;
import numpy as np;
from train import trainer;
from data import dataloader;
from model import V_theta;
from torch.optim.lr_scheduler import StepLR
import os;

    
OPS = {'V':0.1,'E':1,
    'x':0.1, 'y':0.1, 'z':0.1,
    'xx':0.01, 'yy':0.01, 'zz':0.01,
    'xy':0.01, 'yz':0.01, 'xz':0.01,
    'atomic_charge': 0.01, 'E_gap':0.2,
    'bond_order':0.02, 'alpha':3E-5};

device = 'cuda:0';
batch_size = 4;
steps_per_epoch = 1;
N_epoch = 101;
lr_init = 1E-3;
lr_final = 1E-3;
lr_decay_steps = 50;
scaling = {'V':0.2, 'T': 0.01};
Nsave = 50;

element_list = ['H','C','N','O','F'];
path = '/pscratch/sd/t/th1543/v2.0/data';
datagroup = 'group2';

loader = dataloader(device=device, element_list = element_list,
                        path = path, batch_size = batch_size, 
                        starting_basis = 'def2-SVP');

data, labels, obs_mats = loader.load_data('group2');
    
train1 = trainer(device, data, labels,
                filename='model.pt',
                op_matrices=obs_mats);

train1.build_irreps(element_list = element_list);
train1.build_model(scaling = scaling);
train1.build_optimizer(lr=lr_init);
train1.build_charge_matrices(data);

if(os.path.exists('model.pt')):
    train1.load('model.pt');

scheduler = StepLR(train1.optim, step_size=lr_decay_steps,
                    gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));

# define loss function and optimizer
with open('loss.txt','w') as file:
    file.write('epoch\t');
    for i in range(len(OPS)):
        file.write(' loss_'+str(list(OPS.keys())[i])+'\t');
    file.write('\n');
    
# forward pass
for i in range(N_epoch):
    
    loss = train1.train(steps=steps_per_epoch,
                        batch_size = batch_size,
                        op_names=OPS);
    
    scheduler.step();

    with open('loss.txt','a') as file:
        file.write(str(i)+'\t')
        for j in range(len(loss)):
            file.write(str(loss[j])+'\t');
        file.write('\n');

    if(i%Nsave == 0 and i>0):
        train1.save(str(i)+'_model.pt');
        print('saved model at epoch '+str(i));


