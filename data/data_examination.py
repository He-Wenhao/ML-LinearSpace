import json;
import os;
import numpy as np;

filelist = ['group'+str(i) for i in range(14)];

for file in filelist:

    print('Checking', file);
    fl = os.listdir(file+'/basic/');

    for name in fl:

        with open(file + '/basic/' + name) as f:
            basic = json.load(f);
        
        crit1 = np.prod([len(basic[key])!=0 for key in basic if(key!='Enn' and key!='HF')]);
        if(crit1 == 0):
            print(file, name, 'failed basic check');

        with open(file + '/obs/' + name.split('_')[0]+'.json') as f:
            obs = json.load(f);
        crit2 = (len(obs['atomic_charge'])!=0) * \
                (len(obs['bond_order'])!=0) * \
                (len(obs['alpha'])!=0) * \
                (len(obs['T'])!=0);
        
        if(crit2 == 0):
            print(file, name, 'failed obs check');
        