# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:44:30 2023

@author: 17000
"""

import json;
import torch;

class integrate():

    def __init__(self, device):

        with open('script/orbitals.json','r') as file:
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
    
    # to class, read document when initialize
    def calc_F(self, data_in, c, ngrid = 20):
        
        # N: number of atoms, M: number of basis, ne: number of occupied orbitals
        # give pos: Nx3 tensor atomic coordinates and elements: N-dim species list ['C','H', ...]
        # and the MxM tensor wave function vectors c from the K-S equation.
        # This function outputs an ne x (M-ne) Pytorch tensor that output
        # ci * (sum_j ci*F*cj) * ck for all (i,k) pairs
        
        ############ read orbital information into all_orbs ###############
        
        angstron2Bohr = 1.88973;
        all_orbs = [];
        pos = data_in['pos']*angstron2Bohr;
        map1 = [];
        map_ind = 0;        
        
        for atom in data_in['elements']:
            all_orbs += self.orbs_elements[atom];
            if(atom=='H'):
                map1 += [map_ind]*14;
            else:
                map1 += [map_ind]*30;
            map_ind += 1;
                
        ############ calculate all orbitals on grid (psi_all) #############
        norbs = len(all_orbs);
        nbatch = data_in['properties']['nframe'];
        ne = data_in['properties']['ne'];

        crd_min, crd_max = torch.min(pos,axis=1)[0],torch.max(pos,axis=1)[0];

        xx = torch.stack([torch.linspace(crd_min[i][0]-5, crd_max[i][0]+5, ngrid) for i in range(nbatch)]).to(self.device);
        yy = torch.stack([torch.linspace(crd_min[i][1]-5, crd_max[i][1]+5, ngrid) for i in range(nbatch)]).to(self.device);
        zz = torch.stack([torch.linspace(crd_min[i][2]-5, crd_max[i][2]+5, ngrid) for i in range(nbatch)]).to(self.device);

        
        psi_all = torch.zeros((norbs, nbatch, ngrid, ngrid, ngrid)).to(self.device);
    
        for iorb in range(norbs):

            orb_tmp = all_orbs[iorb];
            crd = pos[:,map1[iorb],:];
            psi_all[iorb] = self.calc_psi(orb_tmp, crd, xx, yy,zz);
        
        ########### calculate Fik ##########################################
        
        dV = (xx[:,1]-xx[:,0])*(yy[:,1]-yy[:,0])*(zz[:,1]-zz[:,0]); # volume element

        psi_c = torch.einsum('uij,iuklm->juklm',[c,psi_all]); # K-S eigen wave functions
        
        # implement ci * (sum_j ci*F*cj) * ck
        Fm = torch.einsum('julmn,julmn->ulmn',[psi_c[:ne], psi_c[:ne]]);

        Fik = torch.einsum('iuabc,uabc,kuabc,u->uik',[psi_c[:ne], Fm, psi_c[ne:],dV]);
    
        return 2*Fik; # the factor of 2 is from spin
    
    def calc_S_deploy(self, data_in, ngrid=100):
        
        angstron2Bohr = 1.88973;
        all_orbs = [];
        pos = data_in['pos'][:,:,[2,0,1]]*angstron2Bohr;
        map1 = [];
        map_ind = 0;        
        
        for atom in data_in['elements']:
            all_orbs += self.orbs_elements[atom];
            if(atom=='H'):
                map1 += [map_ind]*14;
            else:
                map1 += [map_ind]*30;
            map_ind += 1;
                
        ############ calculate all orbitals on grid (psi_all) #############
        norbs = len(all_orbs);
        nbatch = len(pos);
        
        crd_min, crd_max = torch.min(pos,axis=1)[0],torch.max(pos,axis=1)[0];
        
        xx = torch.stack([torch.linspace(crd_min[i][0]-5, crd_max[i][0]+5, ngrid) for i in range(nbatch)]).to(self.device);
        yy = torch.stack([torch.linspace(crd_min[i][1]-5, crd_max[i][1]+5, ngrid) for i in range(nbatch)]).to(self.device);
        zz = torch.stack([torch.linspace(crd_min[i][2]-5, crd_max[i][2]+5, ngrid) for i in range(nbatch)]).to(self.device);
        
        psi_all = torch.zeros((norbs, nbatch, ngrid, ngrid, ngrid)).to(self.device);
        
        for iorb in range(norbs):
            
            orb_tmp = all_orbs[iorb];
            crd = pos[:,map1[iorb],:];
            psi_all[iorb] = self.calc_psi(orb_tmp, crd, xx, yy,zz);
        ########### calculate Sik ##########################################
        
        dV = (xx[:,1]-xx[:,0])*(yy[:,1]-yy[:,0])*(zz[:,1]-zz[:,0]); # volume element
        Sik = torch.einsum('iulmn,kulmn,u->uik',[psi_all,psi_all, dV]);
        
        return Sik;
    