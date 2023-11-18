import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn import nn



class V_theta(torch.nn.Module):
    
    def __init__(self, device, emb_neurons: int = 16) -> None:
        super().__init__()
        
        # Initialize a Equivariance graph convolutional neural network
        # nodes with distance smaller than max_radius are connected by bonds
        # num_basis is the number of basis for edge feature embedding
        
        self.device = device;
        
        self.Irreps_HH = [["4x0e","2x1o"],
                         ["2x1o","1x0e+1x1e+1x2e"]];
        
        self.Irreps_CC = [["9x0e","6x1o","3x2e"],                         
                         ["6x1o",("1x0e+1x1e+1x2e+"*4)[:-1],
                          ("1x1o+1x2o+1x3o+"*2)[:-1]],           
                         ["3x2e",("1x1o+1x2o+1x3o+"*2)[:-1],
                          "1x0e+1x1e+1x2e+1x3e+1x4e" 
                          ]];
        
        self.Irreps_CH = [["6x0e","3x1o"],                         
                         ["4x1o",("1x0e+1x1e+1x2e+"*2)[:-1]],           
                         ["2x2e","1x1o+1x2o+1x3o"]];
        
        out = "";
        for f in self.Irreps_HH:
            for f1 in f:
                out += f1+'+';
        out = out+out;
        self.Irreps_HH = o3.Irreps(out[:-1]);
        out = "";
        for f in self.Irreps_CC:
            for f1 in f:
                out += f1+'+';
        out = out+out;
        self.Irreps_CC = o3.Irreps(out[:-1]);
        out = "";
        for f in self.Irreps_CH:
            for f1 in f:
                out += f1+'+';
        out = out+out;
        self.Irreps_CH = o3.Irreps(out[:-1]);
        
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2);
        self.irreps_input = o3.Irreps("2x0e");
        irreps_mid1 = o3.Irreps("8x0e + 8x1o + 8x2e");
        irreps_mid2 = o3.Irreps("8x0e + 8x0o + 8x1e + 8x1o + 8x2e + 8x2o");

        self.tp1 = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_input,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid1,
            shared_weights=False
        )
        self.tp2 = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid1,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid2,
            shared_weights=False
        )
        
        self.bond_feature = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=irreps_mid2,
            irreps_out=irreps_mid2,
            shared_weights=False
        )
        
        self.tpHH = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=self.irreps_sh,
            irreps_out=self.Irreps_HH,
            shared_weights=False
        )
        
        self.tpCC = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=self.irreps_sh,
            irreps_out=self.Irreps_CC,
            shared_weights=False
        )
        
        self.tpCH = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=self.irreps_sh,
            irreps_out=self.Irreps_CH,
            shared_weights=False
        )
        
        self.tpC = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=self.irreps_sh,
            irreps_out=self.Irreps_CC,
            shared_weights=False
        )
        self.tpH = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=self.irreps_sh,
            irreps_out=self.Irreps_HH,
            shared_weights=False
        )
        
        self.fc1 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp1.weight_numel], torch.relu);
        self.fc2 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp2.weight_numel], torch.relu);
        self.fc_bond = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.bond_feature.weight_numel], torch.relu);

        self.fcHH = nn.FullyConnectedNet([1, emb_neurons,emb_neurons,emb_neurons, self.tpHH.weight_numel], torch.relu);
        self.fcCC = nn.FullyConnectedNet([1, emb_neurons,emb_neurons,emb_neurons, self.tpCC.weight_numel], torch.relu);
        self.fcCH = nn.FullyConnectedNet([1, emb_neurons,emb_neurons,emb_neurons, self.tpCH.weight_numel], torch.relu);
        self.fcC = nn.FullyConnectedNet([1, emb_neurons,emb_neurons,emb_neurons, self.tpC.weight_numel], torch.relu);
        self.fcH = nn.FullyConnectedNet([1, emb_neurons,emb_neurons,emb_neurons, self.tpH.weight_numel], torch.relu);
        
    def forward(self, minibatch) -> torch.Tensor:
        
        # Forward function of the neural network model
        # positions is a Nx3 tensor, including the N times 3D atomic cartesian coordinate
        # elements is a N-dim list, whose ith component is either 'H' or 'C' denoting 
        # the ith atomic species.
        # The output is a (N*14)x(N*14) V_theta matrix.
        # The i*14+j row/column means the ith atom's jth basis orbital
        # The basis on an atom is ranked as (s1,s2,s3,p1_(-1,0,1),p2_(-1,0,1),d1_(-2,-1,0,1,2))
        sh = minibatch['sh'];
        emb = minibatch['emb'];
        f_in = minibatch['f_in'];
        edge_src = minibatch['edge_src'];
        edge_dst = minibatch['edge_dst'];
        num_nodes = minibatch['num_nodes'];
        num_neighbors = minibatch['num_neighbors'];
        HH_ind = minibatch['HH_ind'];
        CC_ind = minibatch['CC_ind'];
        CH_ind = minibatch['CH_ind'];
        edge_feature = self.tp1(f_in[edge_src], sh, self.fc1(emb));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        edge_feature = self.tp2(node_feature[edge_src], sh, self.fc2(emb));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        
        HH_feature = self.bond_feature(node_feature[edge_src[HH_ind]], node_feature[edge_dst[HH_ind]],self.fc_bond(emb[HH_ind]));
        CC_feature = self.bond_feature(node_feature[edge_src[CC_ind]], node_feature[edge_dst[CC_ind]],self.fc_bond(emb[CC_ind]));
        CH_feature = self.bond_feature(node_feature[edge_src[CH_ind]], node_feature[edge_dst[CH_ind]],self.fc_bond(emb[CH_ind]));
        
        edge_HH = self.tpHH(HH_feature, sh[HH_ind], self.fcHH(emb[HH_ind]));
        edge_CC = self.tpCC(CC_feature, sh[CC_ind], self.fcCC(emb[CC_ind]));
        edge_CH = self.tpCH(CH_feature, sh[CH_ind], self.fcCH(emb[CH_ind]));
        
        edge_C = self.tpC(node_feature[edge_src], sh, self.fcC(emb));
        node_C = scatter(edge_C, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        edge_H = self.tpH(node_feature[edge_src], sh, self.fcH(emb));
        node_H = scatter(edge_H, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        
        def slice(input,part):
            assert part in [0,1]
            if part == 0:
                return input[:,:int(input.shape[1])//2]
            if part == 1:
                return input[:,int(input.shape[1])//2:]
            
        V_raw0 = {'H': slice(node_H,0),
                 'C': slice(node_C,0),
                 'HH': slice(edge_HH,0),
                 'CH': slice(edge_CH,0),
                 'CC': slice(edge_CC,0)};
        
        V_raw1 = {'H': slice(node_H,1),
                 'C': slice(node_C,1),
                 'HH': slice(edge_HH,1),
                 'CH': slice(edge_CH,1),
                 'CC': slice(edge_CC,1)};
        
        return V_raw0, V_raw1;


