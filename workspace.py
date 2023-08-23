import torch;
import json;
import numpy as np;
from pkgs.train import trainer;
from pkgs.dataframe import load_data;
from pkgs.model import V_theta;

device = 'cpu';
data, labels = load_data('Methane', device,ind_list=[1,2,3]);

model = V_theta(device);

train1 = trainer(device, kn=0, lr=0.001);
train1.optim = torch.optim.Adam(train1.model.parameters(), lr=0.01);
coef = {'V':1/3,'E':1000/3,'n':0.01/3};

for i in range(100):
    
    loss = train1.pretrain_subspace(data, labels, norbs=10, kE=0,steps=10);
    print(loss);

train1.optim = torch.optim.Adam(train1.model.parameters(), lr=1E-4);
for i in range(100):
    
    loss = train1.pretrain_subspace(data, labels,kE=100,steps=10);
    print(loss);

    torch.save(train1.model.state_dict(),'model.pt');
    
train1.optim = torch.optim.Adam(train1.model.parameters(), lr=1E-5);
for i in range(100):
    
    loss = train1.finetune(data, labels,coef,steps=10);
    print(loss);
