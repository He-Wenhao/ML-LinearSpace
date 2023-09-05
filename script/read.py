import os;
import numpy as np;
import json;
from ase.io.cube import read_cube_data;
import torch;
import scipy;

class integrate():

    def __init__(self, device):

        with open('orbitals.json','r') as file:
            self.orbs_elements = json.load(file);
        self.device = device;

    def calc_psi(self, orb, crd, xx, yy, zz):

        # This function calculate the basis wave functions on grid
        # orbs is a dictionary containing N Gaussian function informations  
        # 'c': N-dim weight list of Gaussian functions, 
        # 'alpha': Nx3 list of x^a1 y^a2 z^a3 polynomial exponents
        # 'zeta': N-dim exponential term e^(-zeta r^2);
        # xx, yy, zz are Nx, Ny, Nz grid tensor referenced to the atom center.
        # output psi on Nx x Ny x Nz PyTorch tensor.
        c = orb['c'];
        alpha = orb['alpha'];
        zeta = orb['zeta'];
        nframe = len(crd);
        nx,ny,nz = len(xx[0]), len(yy[0]),len(zz[0]);

        dx = xx.reshape(nframe,nx)-crd[:,0].reshape(nframe,1);
        dy = yy.reshape(nframe,ny)-crd[:,1].reshape(nframe,1);
        dz = zz.reshape(nframe,nz)-crd[:,2].reshape(nframe,1);
        psi = torch.zeros([nframe, nx, ny, nz]).to(self.device);
        for ii in range(len(c)):
            x_components = dx**alpha[ii][0]*torch.exp(-zeta[ii]*dx**2);
            y_components = dy**alpha[ii][1]*torch.exp(-zeta[ii]*dy**2);
            z_components = dz**alpha[ii][2]*torch.exp(-zeta[ii]*dz**2);
            outer_product = torch.einsum('ui,uj,uk->uijk',[x_components,y_components,z_components]);
            psi += c[ii]*outer_product;
        return psi;
    # to class, read document when initialize

    def calc_F(self, data, elements, c, ngrid = 20):

        # N: number of atoms, M: number of basis, ne: number of occupied orbitals
        # give pos: Nx3 tensor atomic coordinates and elements: N-dim species list ['C','H', ...]
        # and the MxM tensor wave function vectors c from the K-S equation.
        # This function outputs an ne x (M-ne) Pytorch tensor that output
        # ci * (sum_j ci*F*cj) * ck for all (i,k) pairs
        ############ read orbital information into all_orbs ###############
        angstron2Bohr = 1.88973;
        all_orbs = [];
        batch =  data.batch;
        pos = data.pos*angstron2Bohr;
        for i in range(len(batch)):
            if(batch[i]==0):
                all_orbs += self.orbs_elements[elements[i]];
            else:
                break;

        ############ calculate all orbitals on grid (psi_all) #############
        crd_min, crd_max = torch.min(pos,axis=0)[0],torch.max(pos,axis=0)[0];
        xx = torch.linspace(crd_min[0]-5, crd_max[0]+5, ngrid).to(self.device);
        yy = torch.linspace(crd_min[1]-5, crd_max[1]+5, ngrid).to(self.device);
        zz = torch.linspace(crd_min[2]-5, crd_max[2]+5, ngrid).to(self.device);
        norbs = len(all_orbs);
        nbatch = torch.max(batch)+1;
        pos_e = pos[[2*i for i in range(nbatch)]].to(self.device);
        pos_o = pos[[2*i+1 for i in range(nbatch)]].to(self.device);
        ne = 1; #int(round(sum([(ele=='C')*5+1 for ele in elements])/2));
        psi_all = torch.zeros((norbs, nbatch, ngrid, ngrid, ngrid)).to(self.device);
        for iorb in range(norbs):
            orb_tmp = all_orbs[iorb];
            if(iorb<5):
                crd = pos_e;
            else:
                crd = pos_o;

            psi_all[iorb] = self.calc_psi(orb_tmp, crd, xx, yy,zz);
        ########### calculate Fik ##########################################

        dV = (xx[1]-xx[0])*(yy[1]-yy[0])*(zz[1]-zz[0]); # volume element
        psi_c = torch.einsum('uij,iuklm->juklm',[c,psi_all]); # K-S eigen wave functions
        # implement ci * (sum_j ci*F*cj) * ck
        Fm = torch.einsum('julmn,julmn->ulmn',[psi_c[:ne], psi_c[:ne]]);
        Fik = torch.einsum('iuabc,uabc,kuabc->uik',[psi_c[:ne], Fm, psi_c[ne:]]);  

        return 2*Fik*dV; # the factor of 2 is fro

    def calc_S(self, pos, atm, grid):
        angstron2Bohr = 1.88973;
        all_orbs = [];
        pos = torch.tensor(pos).to(self.device)*angstron2Bohr;
        map1 = [];
        map_ind = 0;        
        for atom in atm[0]:
            all_orbs += self.orbs_elements[atom];
            if(atom=='H'):
                map1 += [map_ind]*14;
            else:
                map1 += [map_ind]*30;
            map_ind += 1;

        ############ calculate all orbitals on grid (psi_all) #############

        xx = torch.stack([torch.linspace(gr[0][0], gr[0][0]+gr[1][0]*59, 60) for gr in grid]).to(self.device);
        yy = torch.stack([torch.linspace(gr[0][1], gr[0][1]+gr[1][1]*59, 60).to(self.device) for gr in grid]).to(self.device);
        zz = torch.stack([torch.linspace(gr[0][2], gr[0][2]+gr[1][2]*59, 60).to(self.device) for gr in grid]).to(self.device);
        norbs = len(all_orbs);
        nbatch = len(pos);
        psi_all = torch.zeros((norbs, nbatch, 60, 60, 60)).to(self.device);
        for iorb in range(norbs):

            orb_tmp = all_orbs[iorb];
            crd = pos[:,map1[iorb],:];
            psi_all[iorb] = self.calc_psi(orb_tmp, crd, xx, yy,zz);
        ########### calculate Sik ##########################################
        dV = (xx[:,1]-xx[:,0])*(yy[:,1]-yy[:,0])*(zz[:,1]-zz[:,0]); # volume element
        Sik = torch.einsum('iulmn,kulmn,u->uik',[psi_all,psi_all, dV]);

        return Sik;



    def calc_N(self, pos, atm, nr, grid):
        angstron2Bohr = 1.88973;
        all_orbs = [];
        pos = torch.tensor(pos).to(self.device)*angstron2Bohr;
        map1 = [];
        map_ind = 0;        
        for atom in atm[0]:
            all_orbs += self.orbs_elements[atom];
            if(atom=='H'):
                map1 += [map_ind]*14;
            else:
                map1 += [map_ind]*30;
            map_ind += 1;

        ############ calculate all orbitals on grid (psi_all) #############

        xx = torch.stack([torch.linspace(gr[0][0], gr[0][0]+gr[1][0]*59, 60) for gr in grid]).to(self.device);
        yy = torch.stack([torch.linspace(gr[0][1], gr[0][1]+gr[1][1]*59, 60).to(self.device) for gr in grid]).to(self.device);
        zz = torch.stack([torch.linspace(gr[0][2], gr[0][2]+gr[1][2]*59, 60).to(self.device) for gr in grid]).to(self.device);
        norbs = len(all_orbs);
        nbatch = len(pos);
        psi_all = torch.zeros((norbs, nbatch, 60, 60, 60)).to(self.device);
        for iorb in range(norbs):

            orb_tmp = all_orbs[iorb];
            crd = pos[:,map1[iorb],:];
            psi_all[iorb] = self.calc_psi(orb_tmp, crd, xx, yy,zz);
        ########### calculate Sik ##########################################

        nr = torch.tensor(nr).to(self.device);
        dV = (xx[:,1]-xx[:,0])*(yy[:,1]-yy[:,0])*(zz[:,1]-zz[:,0]); # volume element
        Nik = torch.einsum('iulmn,ulmn,kulmn,u->uik',[psi_all,nr,psi_all, dV]);
        return Nik;


Nframe = 500;
Elist = [];
posl  = [];
atm   = [];
nr = [];
grid = [];

route = os.getcwd()+'/';

for i in range(Nframe):
    print(i);
    res = os.popen("grep 'E(TOT)' "+str(i)+'/log').readline();
    if(len(res)!=0):
        E1 = float(res.split()[-1][:-1]);
    else:
        E1 = False;
    Elist.append(E1);

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
for i in range(Nframe):
    with open('PVDZ/methane/'+str(i)+'/log', 'r') as file:
        output =  file.readlines();
        u =  0;
    
        while('**** Energy Check signals convergence ****' not in output[u]):
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
    
    h_raw = readmat(dataf);
    S = readmat(dataS);
    S = scipy.linalg.fractional_matrix_power(S,-1/2);

    h = np.matmul(np.matmul(S, h_raw),S);

    slist.append(S.tolist());
    mlist.append(h.tolist());

integrator = integrate('cpu');
Nik = [];
Sik = [];

for i in range(50):
    Nik += integrator.calc_N(posl[10*i:10*i+10],atm[10*i:10*i+10],nr[10*i:10*i+10],grid[10*i:10*i+10]).tolist();

output = {'coordinates':posl, 'elements':atm, 'S':slist, 'h':mlist, 'energy':Elist, 'N':Nik};

with open('data.json','w') as file:
    json.dump(output, file);

