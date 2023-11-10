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
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    
    batch_size = 500;
    steps_per_epoch = 10;
    N_epoch = 1000;
    k_E = 0.9;
    lr_init = 5E-3;
    lr_final = 1E-5;
    lr_decay_steps = 50;
    
    molecule_list = [['methane','cyclopropane'],['ethane','propyne'],
                     ['propane','acetylene'],['ethylene','propylene']];

    data, labels = load_data(molecule_list[rank], rank, ind_list=[]);

    train1 = trainer_ddp(rank, data, labels, lr=lr_init,
                         filename=str(rank)+'_model.pt');
    
    scheduler = StepLR(train1.optim, step_size=lr_decay_steps,
                       gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));
    
    # define loss function and optimizer
    with open(str(rank)+'.txt','w') as file:
        file.write('epoch\t loss_V\t loss_E\n');
        
    # forward pass
    for i in range(N_epoch):
        
        loss = train1.train(steps=steps_per_epoch, kE = k_E,
                            batch_size = batch_size);
        
        scheduler.step();
        with open(str(rank)+'.txt','a') as file:
            file.write(str(i)+'\t'+str(loss[0])+'\t'+str(loss[1])+'\n');
        
def main():
    world_size = 4
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
    

