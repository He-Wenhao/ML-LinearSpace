import time
import torch;
import json;
import numpy as np;
from pkgs.train import trainer;
from pkgs.dataframe import load_data;
from pkgs.model import V_theta;
from torch.optim.lr_scheduler import StepLR;

OPS = {'V':0.1,'E':1,
       'x':0.1, 'y':0.1, 'z':0.1,
       'xx':0.1, 'yy':0.1, 'zz':0.1,
       'xy':0.1, 'yz':0.1, 'xz':0.1}

device = 'cuda:0';
molecule_list = ['methane','ethane','acetylene','ethylene',
                 'propane','propylene','cyclopropane','propyne',
                 'cyclobutane','cyclopentane','butane','neopentane',
                 '2m2b','benzene','isobutane','isobutylene',
                 '1butyne','2butyne','13butadiene','12butadiene'];
batch_size = 240;
steps_per_epoch = 1;
N_epoch = 1000;
lr_init = 1E-2;
lr_final = 1E-3;
lr_decay_steps = 200;

start_time = time.time()
def timing(msg):
    print(f"{msg} | time from start {time.time() - start_time:.1f} s")

np.set_printoptions(linewidth=np.inf);

print("=" * 75)
print("PARAMETERS:")
print("OPS               ", OPS)
print("device            ", device)
print("molecule_list     ", molecule_list)
print("batch_size        ", batch_size)
print("steps_per_epoch   ", steps_per_epoch)
print("N_epoch           ", N_epoch)
print("lr_init           ", lr_init)
print("lr_final          ", lr_final)
print("lr_decay_steps    ", lr_decay_steps)
print("=" * 75)

operators_except_EV = [key for key in list(OPS.keys()) if key!='E' and key!='V'];
data, labels, obs_mats = load_data(molecule_list, device, 
                                   ind_list=[2*i for i in range(batch_size)], 
                                   op_names=operators_except_EV);

timing("data loaded")

train1 = trainer(device,data, labels, lr=lr_init,
                    op_matrices=obs_mats);
schedular = StepLR(train1.optim, step_size=lr_decay_steps,
                   gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));

Loss = [];
for i in range(N_epoch):

    loss = train1.train(steps=steps_per_epoch,
                        batch_size = batch_size,
                        op_names=OPS);
    print(loss);
    Loss.append(loss.tolist());
    schedular.step();
    timing(f"epoch {i} done")

timing("training complete")

with open('loss.json','w') as file:
    json.dump(Loss,file);
