from train_func import train_func

params = {};

params['OPS'] = {'V':0.1,'E':1,
    'x':0.1, 'y':0.1, 'z':0.1,
    'xx':0.01, 'yy':0.01, 'zz':0.01,
    'xy':0.01, 'yz':0.01, 'xz':0.01,
    'atomic_charge': 0.01, 'E_gap':0.2,
    'bond_order':0.02, 'alpha':3E-5};

params['device'] = 'cuda:0';
params['batch_size'] = 100;
params['steps_per_epoch'] = 10;
params['N_epoch'] = 201;
params['lr_init'] = 1E-3;
params['lr_final'] = 1E-3;
params['lr_decay_steps'] = 50;
params['scaling'] = {'V':1, 'T': 0.01};
params['Nsave'] = 20;

params['element_list'] = ['H','C','N','O','F'];
params['path'] = '/pscratch/sd/t/th1543/v2.0/';
params['datagroup'] = 'group1';
params['load_model'] = True;
params['world_size'] = 1;
params['output_path'] = '/pscratch/sd/t/th1543/v2.0/output/';

train_func(params);


