# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:50:30 2024

@author: haota
"""

import numpy as np;
import json;
import matplotlib.pyplot as plt;
import matplotlib;
from ase import io;
import torch;
from pymatgen.io.vasp import Chgcar, Poscar

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
        nx,ny,nz = len(xx), len(yy),len(zz);

        dx = xx.reshape(nx)-crd[0].reshape(1);
        dy = yy.reshape(ny)-crd[1].reshape(1);
        dz = zz.reshape(nz)-crd[2].reshape(1);
        psi = torch.zeros([nx, ny, nz]).to(self.device);
        
        for ii in range(len(c)):
        
            x_components = dx**alpha[ii][0]*torch.exp(-zeta[ii]*dx**2);
            y_components = dy**alpha[ii][1]*torch.exp(-zeta[ii]*dy**2);
            z_components = dz**alpha[ii][2]*torch.exp(-zeta[ii]*dz**2);
            outer_product = torch.einsum('i,j,k->ijk',[x_components,y_components,z_components]);
            psi += c[ii]*outer_product;
        
        return psi;
    
    # to class, read document when initialize
    
    
    def calc_eigen(self, wave, pos, atm, grid):
        
        angstron2Bohr = 1.88973;
        all_orbs = [];
        pos = torch.tensor(pos).to(self.device)*angstron2Bohr;
        grid = np.array(grid)*angstron2Bohr;
        map1 = [];
        map_ind = 0;        
        dic = {1:'H',6:'C'}
        for atom in atm:
            
            all_orbs += self.orbs_elements[dic[atom]];
            if(atom==1):
                map1 += [map_ind]*5;
            else:
                map1 += [map_ind]*14;
            map_ind += 1;

        ############ calculate all orbitals on grid (psi_all) #############

        xx = torch.linspace(grid[0][0], grid[1][0], int(grid[2][0])).to(self.device);
        yy = torch.linspace(grid[0][1], grid[1][1], int(grid[2][1])).to(self.device);
        zz = torch.linspace(grid[0][2], grid[1][2], int(grid[2][2])).to(self.device);
        norbs = len(all_orbs);
        psi_all = torch.zeros(norbs, int(grid[2][0]), 
                              int(grid[2][1]), int(grid[2][2])).to(self.device);
        
        for iorb in range(norbs):

            orb_tmp = all_orbs[iorb];
            crd = pos[map1[iorb],:];
            psi_all[iorb] = self.calc_psi(orb_tmp, crd, xx, yy,zz)*angstron2Bohr**(3/2);
        ########### calculate Sik ##########################################
        
        psi = torch.einsum('ilmn, i->lmn',[psi_all, wave]);

        return psi;

Nl  = [{'file':'PA', 'N':19, 'NC':38, 'NH':40},
      {'file':'benzene', 'N':10, 'NC':60, 'NH':42},
      {'file':'cyclic', 'N':6, 'NC':66, 'NH':66}];
N = Nl[2];

device = 'cuda:0';

with open('../deploy/deploy/test_output/C'+str(N['NC'])+'H'+str(N['NH'])+'.json','r') as file:
    data = json.load(file);

atomic_charge = list(data['atomic_charge'].values())[0][0];

atoms = io.read('../deploy/' + N['file'] + '/polymer/POSCAR_'+str(N['N']));

atm = atoms.get_atomic_numbers();
pos = atoms.get_positions();

font = {
'size' : 32}

matplotlib.rc('font', **font)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.figure(figsize=(10,8))  

with open('H_' + N['file'] + '.json','r') as file:
    H = np.array(json.load(file))[0];
ne = (N['NC']*6+N['NH'])//2
energy, wave = np.linalg.eigh(H);

HOMO = torch.tensor(wave[:, ne-1], dtype=torch.float).to(device);
left = np.zeros(3);
right = np.diag(atoms.cell)
grid = np.stack([left,
                 right,
                 np.floor((right-left)/0.4)]).tolist()
integrator = integrate(device);
psi = integrator.calc_eigen(HOMO, pos, atm, grid)
volume = np.prod(right-left);

poscar = Poscar.from_file('../deploy/' + N['file'] + '/polymer/POSCAR_'+str(N['N']), format='poscar')
data  = {'total': np.array(psi.tolist())*volume}
chg = Chgcar(poscar, data);
chg.write_file('CHGCAR_'+N['file'])