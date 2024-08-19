# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:44:30 2023

@author: 17000
"""

import json;
import torch;
from scipy.interpolate import lagrange;
import numpy as np;
import scipy;
from periodictable import elements
from e3nn.o3 import spherical_harmonics as sh;

class integrate():

    def __init__(self, device, starting_basis, path, lmaxf = 4):

        if(starting_basis == 'cc-pVDZ'):
            filename = path + 'script/orbitals_CH.json';
        elif(starting_basis == 'def2-SVP'):
            filename = path +'script/orbitals_new.json';
        else:
            raise ValueError('The basis set is not supported');
            
        with open(filename, 'r') as file:
            self.orbs_elements = json.load(file);
        self.device = device;
        self.zeta = {};
        self.alpha = {};
        self.c = {};
        self.Ng = {};
        self.Nb = {};
        
        for key in self.orbs_elements:
            self.zeta[key] = [];
            self.alpha[key] = [];
            self.c[key] = [];
            self.Ng[key] = 0;
            self.Nb[key] = 0;
            
            for u in self.orbs_elements[key]:
                self.zeta[key] += u['zeta'];
                self.alpha[key] += u['alpha'];
                self.c[key].append(u['c']);
                self.Ng[key] += len(u['c']);
                self.Nb[key] += 1;
        
        xlist = np.linspace(-4,4,9);
        matrix = [];
        for i in range(9):
            ylist = np.zeros(9);
            ylist[i] = 1;
            res = np.flip(lagrange(xlist, ylist).coef);
            res = res.tolist()+[0]*(9-len(res));
            matrix.append([res[2*n]*scipy.special.factorial2(max(2*n-1,0))/2**n*np.sqrt(np.pi) for n in range(5)]);
        self.mat = torch.tensor(matrix, dtype=torch.float).to(self.device);

        self.lmaxf = lmaxf;
        Ngrid = int((lmaxf)/2)+1;
        r = torch.linspace(-Ngrid, Ngrid, 2*Ngrid+1).to(self.device);
        x,y,z = r[:,None,None], r[None,:,None], r[None,None,:];
        self.r = torch.stack([x + 0*y + 0*z, 0*x + y + 0*z, 0*x + 0*y + z]).reshape([3,-1]);
        rnorm = torch.linalg.norm(self.r, axis=0)[:,None];
        shmat = [];
        for l in range(lmaxf+1):
            shmat.append(sh(l, self.r.T, False));
            if(l<=lmaxf-2):
                shmat.append(sh(l, self.r.T, False)*rnorm**2);
            if(l<=lmaxf-4):
                shmat.append(sh(l, self.r.T, False)*rnorm**4);

        shmat = torch.hstack(shmat).to(self.device);
        self.shmat = torch.linalg.pinv(shmat);
                
    def Mat(self, z1, z2, x1, x2, k1, k2, polynomial):
        
        polynomial = torch.tensor(polynomial,
                                  dtype=torch.float)[None,None,None,:,None].to(self.device);
        xc = (z1[:,:,:,None]*x1+z2[:,:,:,None]*x2)/(z1+z2)[:,:,:,None];
        xc, r1, r2 = xc[:,:,:,:,None],(x1-xc)[:,:,:,:,None], (x2-xc)[:,:,:,:,None];
        exponent_term = torch.exp(-z1*z2/(z1+z2)*torch.sum((x1-x2)**2, axis=3));
        
        x = torch.linspace(-4,4,9)[None,None,None,None,:].to(self.device);
        integrant = (x+xc)**polynomial*(x-r1)**k1[:,:,:,:,None]*(x-r2)**k2[:,:,:,:,None];
        integrant = torch.einsum('ijklm,mn->ijkln', [integrant, self.mat]);
        
        divided = (z1+z2)[:,:,:,None,None]**torch.tensor([1/2+i for i in range(5)],
                                                         dtype=torch.float)[None,None,None,None,:].to(self.device);
        integrant /= divided;
        results = torch.sum(integrant, axis=4);
        results = torch.prod(results, axis=3)*exponent_term;
        
        return results;
    
    def calc_O(self, pos, atm, operator):
        
        polynomial = [operator.count('x'),
                      operator.count('y'),
                      operator.count('z')];
        
        angstron2Bohr = 1.88973;
        pos = pos.to(self.device)*angstron2Bohr;
        pos[:,:,[1,2,0]] = pos[:,:,[0,1,2]];
        mass = {el.symbol: el.mass for el in elements}

        ml = torch.tensor([mass[e1] for e1 in atm], dtype=torch.float).to(self.device);
        center = torch.einsum('uij,i->uj',[pos,ml])/torch.sum(ml);
        pos -= center[:,None,:];
        
        zeta, alpha, c, x = [], [], [], [];
        Nbasis, Ngaussian = 0,0;
        for i in range(len(atm)):
            atom = atm[i];
            zeta += self.zeta[atom];
            alpha += self.alpha[atom];
            x.append(pos[:,i:i+1,:].repeat([1,len(self.zeta[atom]),1]));
                
            c += self.c[atom];
            Ngaussian += self.Ng[atom];
            Nbasis += self.Nb[atom];
            
        x = torch.hstack(x).to(self.device);
        zeta = torch.tensor(zeta, dtype=torch.float).to(self.device);
        alpha = torch.tensor(alpha, dtype=torch.float).to(self.device);

        z1, z2 = zeta[None,:,None], zeta[None,None,:];
        k1, k2 = alpha[None,:,None,:], alpha[None,None,:,:];
        x1, x2 = x[:,:,None,:], x[:,None,:,:];
        
        Omat = self.Mat(z1, z2, x1, x2, k1, k2, polynomial);
        cmat = torch.zeros([Nbasis,Ngaussian], dtype=torch.float).to(self.device);
        
        c_index = 0;
        for i in range(len(c)):
            length = len(c[i]);
            cmat[i, c_index:c_index+length] = torch.tensor(c[i], dtype=torch.float).to(self.device);
            c_index += length;
            
        Omat = torch.einsum('ki,uij,lj->ukl',[cmat, Omat, cmat]);
        
        return Omat.tolist();
    
    def Fmat(self, z1, z2, x1, x2, k1, k2, R, Numint = 40):
        
        lmax = self.lmaxf;
        zeta = z1 + z2;
        zstar = z1*z2/zeta;
        R0 = (z1*x1+z2*x2)/zeta;
        Rd = x1-x2;
        Rp = R - R0;
        Rp.requires_grad = True;
        Rpn = torch.linalg.norm(Rp, axis=3)[:,:,:,None].detach();

        exponent_term = 4*torch.pi*torch.exp(-torch.sum(zstar*Rd**2, axis=3));
        
        r = self.r[None,None,None,...];
        integrant = (r + (R0 - x1)[...,None])**k1[...,None] * (r + (R0 - x2)[...,None])**k2[...,None];
        integrant = torch.prod(integrant, axis=3);

        clm = torch.einsum('ijkl,nl->ijkn', [integrant, self.shmat]);
        rint = torch.linspace(0, 1, Numint).to(self.device);
        rint = (rint[:-1]+rint[1:])[None,None,None,None,:]/2;

        I1, I2, shlist = [], [], [];
        for l in range(lmax+1):
            I1.append(torch.mean(torch.exp(-(zeta*Rpn**2)[...,None]*rint**2) * rint**(2*l+2) * Rpn[...,None]**(2*l+3),
                                axis=4));
            I2.append(torch.exp(-(zeta*Rpn**2))/2/zeta);
            shlist.append(sh(l, Rp, False));
            if(l<=lmax-2):
                I1.append(torch.mean(torch.exp(-(zeta*Rpn**2)[...,None]*rint**2) * rint**(2*l+4) * Rpn[...,None]**(2*l+5),
                                axis=4))
                I2.append(I2[-1]*(1+zeta*Rpn**2)/zeta);
                shlist.append(sh(l, Rp, False));
            if(l<=lmax-4):
                I1.append(torch.mean(torch.exp(-(zeta*Rpn**2)[...,None]*rint**2) * rint**(2*l+6) * Rpn[...,None]**(2*l+7),
                                axis=4));
                I2.append(I2[-2]*Rpn**4+2/zeta*I2[-1]);
                shlist.append(sh(l, Rp, False));

        prod = clm * torch.cat(shlist, dim=3);
        sum1, vec, ind0, ind1 = 0, 0, 0, 0;
        crit = (Rpn < 1e-3);
        for l in range(lmax+1):
            sum1 += torch.sum(prod[...,ind0:ind0+2*l+1] / (2*l+1) * (I1[ind1] / (Rpn+crit)**(2*l+1) + I2[ind1]));
            tmp = torch.sum(prod[...,ind0:ind0+2*l+1] * I1[ind1] / (Rpn+crit)**(2*l+3), axis=3);
            vec += tmp[...,None] * Rp;
            ind0 += 2*l+1;
            ind1 += 1;
            if(l<=lmax-2):
                sum1 += torch.sum(prod[...,ind0:ind0+2*l+1] / (2*l+1) * (I1[ind1] / (Rpn+crit)**(2*l+1) + I2[ind1]));
                tmp = torch.sum(prod[...,ind0:ind0+2*l+1] * I1[ind1] / (Rpn+crit)**(2*l+3), axis=3);
                vec += tmp[...,None] * Rp;
                ind0 += 2*l+1;
                ind1 += 1;
            if(l<=lmax-4):
                sum1 += torch.sum(prod[...,ind0:ind0+2*l+1] / (2*l+1) * (I1[ind1] / (Rpn+crit)**(2*l+1) + I2[ind1]));
                tmp = torch.sum(prod[...,ind0:ind0+2*l+1] * I1[ind1] / (Rpn+crit)**(2*l+3), axis=3);
                vec += tmp[...,None] * Rp;
                ind0 += 2*l+1;
                ind1 += 1;

        sum1.backward();
        gradterm = Rp.grad;
        results = exponent_term[...,None] * (gradterm - vec);

        return results;

    def calc_F(self, pos, atm):
        
        angstron2Bohr = 1.88973;
        pos = pos.to(self.device)*angstron2Bohr;
        pos[:,[1,2,0]] = pos[:,[0,1,2]];
        
        zeta, alpha, c, x = [], [], [], [];
        Nbasis, Ngaussian = 0,0;
        for i in range(len(atm)):
            atom = atm[i];
            zeta += self.zeta[atom];
            alpha += self.alpha[atom];
            x.append(pos[i:i+1,:].repeat([len(self.zeta[atom]),1]));
                
            c += self.c[atom];
            Ngaussian += self.Ng[atom];
            Nbasis += self.Nb[atom];
            
        x = torch.vstack(x).to(self.device);
        zeta = torch.tensor(zeta, dtype=torch.float).to(self.device);
        alpha = torch.tensor(alpha, dtype=torch.float).to(self.device);
        
        # indices: nucleus to calculate force, first gaussian, second gaussian, xyz coordinates
        z1, z2 = zeta[None,:,None,None], zeta[None,None,:,None]; 
        k1, k2 = alpha[None,:,None,:], alpha[None,None,:,:];
        x1, x2 = x[None,:,None,:], x[None,None,:,:];
        R = pos[:,None,None,:];

        Omat = self.Fmat(z1, z2, x1, x2, k1, k2, R);
        cmat = torch.zeros([Nbasis,Ngaussian], dtype=torch.float).to(self.device);
        
        c_index = 0;
        for i in range(len(c)):
            length = len(c[i]);
            cmat[i, c_index:c_index+length] = torch.tensor(c[i], dtype=torch.float).to(self.device);
            c_index += length;

        Omat = torch.einsum('ki,uijv,lj->uvkl',[cmat, Omat, cmat]);
        Omat[:,[0,1,2],:,:] = Omat[:,[1,2,0],:,:];

        return Omat.tolist();