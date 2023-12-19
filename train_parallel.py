import torch;
import json;
import numpy as np;
from pkgs.train import trainer_ddp;
from pkgs.dataframe import load_data;
from pkgs.model import V_theta;
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
import os;

def example(rank, world_size):
    # create default process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    
    OPS = {'V':0.1,'E':1,
           'x':0.1, 'y':0.1, 'z':0.1,
           'xx':0.05, 'yy':0.05, 'zz':0.05,
           'xy':0.1, 'yz':0.1, 'xz':0.1}

    batch_size = 500;
    steps_per_epoch = 1;
    N_epoch = 1000;
    lr_init = 1E-2;
    lr_final = 1E-3;
    lr_decay_steps = 50;
    
    molecule_list = [['methane','cyclopropane'],['ethane','propyne'],
                     ['propane','acetylene'],['ethylene','propylene']];

    molecule_list = [['methane'],['acetylene']];

    operators_except_EV = [key for key in list(OPS.keys()) if key!='E' and key!='V'];
    data, labels, obs_mats = load_data(molecule_list[rank], rank, 
                                       ind_list=range(batch_size), 
                                       op_names=operators_except_EV);

    train1 = trainer_ddp(rank, data, labels, lr=lr_init,
                         filename=str(rank)+'_model.pt',
                         op_matrices=obs_mats);
    
    scheduler = StepLR(train1.optim, step_size=lr_decay_steps,
                       gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));
    
    # define loss function and optimizer
    with open(str(rank)+'.txt','w') as file:
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
        with open(str(rank)+'.txt','a') as file:
            file.write(str(i)+'\t')
            for i in range(len(loss)):
                file.write(str(loss[i])+'\t');
            file.write('\n');
        
def main():
    world_size = 2;
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    main()
    

