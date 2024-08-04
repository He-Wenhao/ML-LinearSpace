from basis.integral import integrate
from model.model_cls import V_theta
from deploy.predictor import predict_fns
from model.sample_minibatch import sampler as sample_train
from data.loader import dataloader
import numpy as np
import torch
from model.tomat import to_mat;
import os;
import json;
import matplotlib;
import matplotlib.pyplot as plt;
import scipy;
from torch_cluster import radius_graph;
from e3nn import o3;
from deploy.deploy_cls import estimator_test
from periodictable import elements

class estimator(estimator_test):

    def __init__(self, device, data_in, op_matrices, 
                 filename, output_folder='output/inference') -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        self.filename = filename;
        self.sampler = sampler(data_in, device);
        self.n_molecules = len(data_in);
        self.op_matrices = op_matrices;
        self.el_dict = {el.symbol: {'Z':el.number,'M':el.mass} for el in elements};

        if(not os.path.exists(output_folder)):
            os.mkdir(output_folder);

        return None;

    def get_nuclear_charge(self):

        elements = self.sampler.data_in['elements'];
        nuclearCharge = [self.el_dict[ele]['Z'] for ele in elements];
        self.nuclearCharge = torch.tensor(nuclearCharge, dtype=torch.float).to(self.device);

        return self.nuclearCharge;

    def get_centered_pos(self):

        angstron2Bohr = 1.88973;
        pos = self.sampler.data_in['pos'];
        elements = self.sampler.data_in['elements'];
        mass = torch.tensor([self.el_dict[ele]['M'] for ele in elements], 
                            dtype=torch.float).to(self.device);
        mass_center = torch.sum(pos*mass[None,:,None], axis=1)/torch.sum(mass);
        pos = (pos - mass_center[:,None,:])*angstron2Bohr;

        return pos;
    
    def get_multipole(self, op_name):

        pos = self.get_centered_pos();
        nuclearCharge = self.get_nuclear_charge();
        moment = torch.tensor([op_name.count('x'),
                               op_name.count('y'),
                               op_name.count('z')], dtype=torch.float).to(self.device);
        multipole = torch.sum(torch.prod(pos**moment[None,None,:],
                                         axis=2)*nuclearCharge[None,:], axis=1);

        return multipole;


    def solve_apply(self, batch_size=1, op_names=[]) -> dict:
        
        properties = self.solve(batch_size, op_names);

        obj = {};
        obj['C'] = elements.count('C');
        obj['H'] = elements.count('H');
        
        for i, op_name in enumerate(op_names):

            if(op_name in ['x','y','z','xx','yy','zz','xy','xz','yz']):

                multipole = self.get_multipole(op_name);                 
                O_mat = obs_mat[op_name]
                Ohat = pred.O(O_mat)
                Ohat = multipole - Ohat;
                obj[op_name] = {'Ohat': Ohat.tolist()};
        
            if(op_name == 'atomic_charge'):

                Chat = pred.C(Cmat);
                Chat = nuclearCharge[None,:]-Chat;
                obj['atomic_charge'] = {'Chat': Chat.tolist()};

        with open(self.output_folder+'/'+save_filename+'.json','w') as file:
            json.dump(obj,file)

        return obj;

class load_data(dataloader):

    def __init__(self, device, element_list, 
                 path, starting_basis = 'def2-SVP') -> None:

        dataloader.__init__(device=device, element_list = element_list,
                            path = path, batch_size = None, 
                            starting_basis = 'def2-SVP');

    def load(self, filename):

        path_list = [self.path + g + '/basic/' for g in group];
        fl, partition = self.get_files(path_list, 0, 1);
        
        data_in = [];
        obs_mat = [];

        for file in fl:
            
            basic_path = file[0] + file[1];
            data_in.append(self.read_basic(basic_path));
            obs_mat.append(self.read_obs_mat());

        return data_in, obs_mat;



class sampler(sample_train):

    def __init__(self, data_in, device) -> None:

        sampler_train.__init__(self, data_in, [], device);
    
    def sample(self, irreps, op_names=[]):

        data = self.data;

        elements = [u['elements'] for u in data];
        natms = [len(u) for u in elements];
        num_nodes = sum(natms);

        pos = torch.vstack([dp['pos'] for dp in data]);
        batch = torch.cat([torch.tensor([i]*n).to(self.device) for i,n in enumerate(natms)]);
        edge_src, edge_dst = radius_graph(x=pos, r=self.max_radius, batch=batch);
        self_edge = torch.tensor([i for i in range(num_nodes)]).to(self.device);
        edge_src = torch.cat((edge_src, self_edge));
        edge_dst = torch.cat((edge_dst, self_edge));
        edge_vec = pos[edge_src] - pos[edge_dst];

        num_neighbors = self.count(batch, batch[edge_src]);

        sh = o3.spherical_harmonics(l = self.irreps_sh, 
                                    x = edge_vec, 
                                    normalize=True, 
                                    normalization='component').to(self.device)

        rnorm = edge_vec.norm(dim=1);
        emb = self.radius_embedding(rnorm);
        f_in = self.element_embedding(elements, irreps);

        pair_ind = [];
        nele = irreps.get_input_irreps();
        for i in range(nele):
            for j in range(i+1):
                ind1 = torch.argwhere(f_in[edge_src][:,i]*f_in[edge_dst][:,j]).reshape(-1);
                pair_ind.append(ind1);

        map1 = self.get_map(elements, irreps);
        basis_max = max([dp['norbs'] for dp in data]);

        minibatch = {
                     'sh': sh,
                     'emb': emb,
                     'f_in':f_in,
                     'edge_src': edge_src,
                     'edge_dst': edge_dst,
                     'num_nodes': num_nodes,
                     'num_neighbors':num_neighbors,
                     'pair_ind': pair_ind,
                     'norbs': [dp['norbs'] for dp in data],
                     'batch': batch,
                     'map1': map1,
                     'natm': natms,
                     'ne': [dp['ne'] for dp in data],
                     'h': self.get_tensor(data, 'h', basis_max)
                     };

        return minibatch;