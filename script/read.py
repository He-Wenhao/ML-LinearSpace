import os;
import numpy as np;
import json;
from ase.io.cube import read_cube_data;
import torch;
import scipy;

Nframe = 500;

Elist = [];
posl  = [];
atm   = [];
nr = [];
grid = [];

E_orb = [];
E_nn = [];
E_HF = [];

route = os.getcwd()+'/';
u = -2;
while(route[u]!='/'):
    u -= 1;
name = route[u+1:-1];

res = os.popen("grep 'Number of Electrons' pvdz/0/log").readline();
ne = float(res.split()[-1]);
print(ne)

os.chdir(route+'pvtz/');
for i in range(Nframe):
    print(i);
    res = os.popen("grep 'E(CCSD(T))' "+str(i)+'/log').readline();
    if(len(res)!=0):
        E1 = float(res.split()[-1][:-1]);
    else:
        E1 = False;
    Elist.append(E1);

    res = os.popen("grep 'SINGLE POINT ENERGY' ../pvdz/"+str(i)+'/log').readline();
    if(len(res)!=0):
        E1 = float(res.split()[-1][:-1]);
    else:
        E1 = False;
    E_HF.append(E1);

    res = os.popen("grep 'One Electron Energy' ../pvdz/"+str(i)+'/log').readline();
    E_orb.append(float(res.split()[-4]));
    res = os.popen("grep 'Nuclear Repulsion' ../pvdz/"+str(i)+'/log').readline();
    E_nn.append(float(res.split()[-2]));

    posl.append([]);
    atm.append([]);
    with open(str(i)+'/run.inp','r') as file:
        data = file.readlines();
        n,test = 0,'';
        while(test != '* xyz'):
            n += 1;
            test = data[-n-2][:5];
        data = data[-1-n:-1];
        for dp in data:
            posl[-1].append([float(u[:-1]) for u in dp.split()[1:]]);
            atm[-1].append(dp.split()[0]); 
                       
    if('run.eldens.cube' in os.listdir(str(i))):
        data, atoms = read_cube_data(str(i)+'/run.eldens.cube');
        data = data[20:80,20:80,20:80].tolist();
        with open(str(i)+'/run.eldens.cube','r') as file:
            res = file.readlines();
            init = [float(x) for x in res[2].split()[1:]];
            dist = [float(res[3].split()[1]), float(res[4].split()[2]), float(res[5].split()[3])];
            init = [init[i]+dist[i]*20 for i in range(3)];
            grid.append([init,dist]);
            nr.append(data)
    else:
        nr.append(False);
        grid.append(False);

def readmat(data):
    number = int(data[-1].split()[0])+1;
    rep = int(round(len(data)/(number+1)));

    matl = [];
    for i in range(rep):
        res = [[float(t) for t in s.split()[1:]] for s in data[i*(number+1)+1:i*(number+1)+number+1]];
        matl.append(np.array(res));
        
    mat = np.hstack(matl)
    return mat;

mlist = [];
slist = [];
os.chdir(route);

for i in range(Nframe):
    print(i)
    with open('pvdz/' + str(i)+'/log', 'r') as file:
        output =  file.readlines();
        u =  0;
        try:
            while('signals convergence' not in output[u]):
                u += 1;
                if('OVERLAP MATRIX' in output[u]):
                    s = int(u)+1;
                if('Time for model grid setup' in output[u]):
                    t = int(u);
            v = int(u);
            while('Fock matrix for operator 0' not in output[v]):
                v -= 1;
            dataf = output[v+1:u];
            dataS = output[s+1:t];

            h = readmat(dataf);
            S = readmat(dataS);

            h += (-np.sum(scipy.linalg.eigvalsh(h,S)[:int(ne/2)])*2 + E_HF[i] - E_nn[i])/ne*S;

            slist.append(S.tolist());
            mlist.append(h.tolist());
        except:
            slist.append(False);
            mlist.append(False);
            print('convergence failure '+str(i));
#integrator = integrate('cpu');
#Nik = [];
Sik = [];

#for i in range(int(Nframe//10)):
#    Nik += integrator.calc_N(posl[10*i:10*i+10],atm[10*i:10*i+10],nr[10*i:10*i+10],grid[10*i:10*i+10]).tolist();

output = {'coordinates':posl, 'HF': E_HF, 'elements':atm, 'S':slist, 'h':mlist, 'energy':Elist, 'Enn':E_nn};

with open(name+'_data.json','w') as file:
    json.dump(output, file);

