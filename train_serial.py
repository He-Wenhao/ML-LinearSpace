import torch;
import json;
import numpy as np;
from pkgs.train import trainer;
from pkgs.dataframe import load_data;
from pkgs.model import V_theta;
from torch.optim.lr_scheduler import StepLR;

#device = 'cuda:0';
device = 'cpu'
molecule_list = ['OH'];
batch_size = 50;
steps_per_epoch = 10;
N_epoch = 1000;
k_E = 0.9;
lr_init = 5E-3;
lr_final = 1E-5;
lr_decay_steps = 50;

training_set = list(range(251))
np.random.shuffle(training_set)
training_set = training_set[:125]

data, labels = load_data(molecule_list, device, ind_list=training_set);

train1 = trainer(device,data, labels, lr=lr_init);
schedular = StepLR(train1.optim, step_size=lr_decay_steps,
                   gamma=(lr_final/lr_init)**(lr_decay_steps/N_epoch));

Loss = [];

for i in range(N_epoch):
    
    loss = train1.train(steps=steps_per_epoch, kE = k_E, batch_size = batch_size);
    print(loss);
    Loss.append(loss.tolist());
    schedular.step();

with open('loss.json','w') as file:
    json.dump(Loss,file);
