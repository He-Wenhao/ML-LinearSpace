import torch;
import json;
import numpy as np;
from torch.optim.lr_scheduler import StepLR;
import os;

import torch.distributed as dist;
from torch.nn.parallel import DistributedDataParallel as DDP;

from train import trainer, recorder;
from data import dataloader;
from model import V_theta;
from basis import read_basis;

############## Setting parameters ################

def train_func(params):

    OPS = params['OPS'];
    device = params['device'];
    batch_size = params['batch_size'];
    steps_per_epoch = params['steps_per_epoch'];
    N_epoch = params['N_epoch'];
    lr_init = params['lr_init'];
    lr_final = params['lr_final'];
    lr_decay_steps = params['lr_decay_steps'];
    scaling = params['scaling'];
    Nsave = params['Nsave'];

    element_list = params['element_list'];
    path = params['path'];
    output_path = params['output_path'];
    datagroup = params['datagroup'];
    load_model = params['load_model'];

    world_size = params['world_size'];

    ###################### load data and model ######################

    if(world_size>1):

        dist.init_process_group('nccl');
        rank = dist.get_rank();
        device = rank % torch.cuda.device_count();

    else:

        rank = 0;

    basis_path = path + 'basis';
    data_path = path + 'data';
    read_basis(path = basis_path, savefile=True);

    loader = dataloader(device=device, element_list = element_list,
                            path = data_path, batch_size = batch_size, 
                            starting_basis = 'def2-SVP');

    data, labels, obs_mats, batch_size = loader.load_data(datagroup, rank, world_size);
        
    train1 = trainer(device, data, labels,
                    filename='model.pt',
                    op_matrices=obs_mats);

    train1.build_irreps(element_list = element_list);
    
    if(world_size>1):
        train1.build_ddp_model(scaling = scaling);
    else:
        train1.build_model(scaling = scaling);
    
    train1.build_optimizer(lr=lr_init);
    train1.build_charge_matrices(data);

    if(load_model):
        train1.load('model.pt');

    scheduler = StepLR(train1.optim, step_size=lr_decay_steps,
                        gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));

    rec = recorder(OPS, rank=rank, path = output_path);
    ############### Implement model training #####################
    for i in range(N_epoch):
        
        loss = train1.train(steps=steps_per_epoch,
                            batch_size = batch_size,
                            op_names=OPS);
        
        scheduler.step();
        rec.record_loss(i, loss);
        rec.save_model(train1, i, Nsave);

    if(world_size>1):
        dist.destroy_process_group()