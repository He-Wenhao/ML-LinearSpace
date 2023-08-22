import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn import nn
from pkgs.tomat import to_mat;

class V_theta(torch.nn.Module):
    
    def __init__(self, device, min_radius: float = 0.5, max_radius: float = 2, emb_neurons: int = 32) -> None:
        super().__init__()
        
        # Initialize a Equivariance graph convolutional neural network
        # nodes with distance smaller than max_radius are connected by bonds
        # num_basis is the number of basis for edge feature embedding
        
        self.max_radius = max_radius;
        self.min_radius = min_radius;
        self.element_embedding = {'H':[0,1],'C':[1,0]};
        self.device = device;
        self.transformer = to_mat(device);
        
        self.Irreps_HH = [["9x0e","6x1o","3x2e"],
                         ["6x1o",("1x0e+1x1e+1x2e+"*4)[:-1],("1x1o+1x2o+1x3o+"*2)[:-1]],
                         ["3x2e",("1x1o+1x2o+1x3o+"*2)[:-1],"1x0e+1x1e+1x2e+1x3e+1x4e"]];
        
        self.Irreps_CC = [["16x0e","12x1o","8x2e","4x3o"],                         
                         ["12x1o",("1x0e+1x1e+1x2e+"*9)[:-1],
                          ("1x1o+1x2o+1x3o+"*6)[:-1],
                          ("1x2e+1x3e+1x4e+"*3)[:-1]],           
                         ["8x2e",("1x1o+1x2o+1x3o+"*6)[:-1],
                          ("1x0e+1x1e+1x2e+1x3e+1x4e+"*4)[:-1], 
                          ("1x1o+1x2o+1x3o+1x4o+1x5o+"*2)[:-1],
                          ],
                         ["4x3o",("1x2e+1x3e+1x4e+"*3)[:-1],
                          ("1x1o+1x2o+1x3o+1x4o+1x5o+"*2)[:-1],
                          "0e+1e+2e+3e+4e+5e+6e"]];
        
        self.Irreps_CH = [["12x0e","8x1o","4x2e"],                         
                         ["9x1o",("1x0e+1x1e+1x2e+"*6)[:-1],
                          ("1x1o+1x2o+1x3o+"*3)[:-1]],           
                         ["6x2e",("1x1o+1x2o+1x3o+"*4)[:-1],
                          ("1x0e+1x1e+1x2e+1x3e+1x4e+"*2)[:-1]
                          ],
                         ["3x3o",("1x2e+1x3e+1x4e+"*2)[:-1],
                          "1x1o+1x2o+1x3o+1x4o+1x5o"]];
        
        out = "";
        for f in self.Irreps_HH:
            for f1 in f:
                out += f1+'+';
        self.Irreps_HH = o3.Irreps(out[:-1]);
        out = "";
        for f in self.Irreps_CC:
            for f1 in f:
                out += f1+'+';
        self.Irreps_CC = o3.Irreps(out[:-1]);
        out = "";
        for f in self.Irreps_CH:
            for f1 in f:
                out += f1+'+';
        self.Irreps_CH = o3.Irreps(out[:-1]);
        
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2);
        self.irreps_input = o3.Irreps("2x0e");
        irreps_mid1 = o3.Irreps("32x0e + 32x1o + 16x2e");
        irreps_mid2 = o3.Irreps("16x0e + 16x1o + 16x2e + 8x3o + 8x4e");

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
        
        self.tpHH = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=irreps_mid2,
            irreps_out=self.Irreps_HH,
            shared_weights=False
        )
        
        self.tpCC = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=irreps_mid2,
            irreps_out=self.Irreps_CC,
            shared_weights=False
        )
        
        self.tpCH = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_mid2,
            irreps_in2=irreps_mid2,
            irreps_out=self.Irreps_CH,
            shared_weights=False
        )
        
        self.fc1 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp1.weight_numel], torch.relu);
        self.fc2 = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tp2.weight_numel], torch.relu);
        self.fcHH = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tpHH.weight_numel], torch.relu);
        self.fcCC = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tpCC.weight_numel], torch.relu);
        self.fcCH = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, self.tpCH.weight_numel], torch.relu);
        
    def forward(self, data_in) -> torch.Tensor:
        
        # Forward function of the neural network model
        # positions is a Nx3 tensor, including the N times 3D atomic cartesian coordinate
        # elements is a N-dim list, whose ith component is either 'H' or 'C' denoting 
        # the ith atomic species.
        # The output is a (N*14)x(N*14) V_theta matrix.
        # The i*14+j row/column means the ith atom's jth basis orbital
        # The basis on an atom is ranked as (s1,s2,s3,p1_(-1,0,1),p2_(-1,0,1),d1_(-2,-1,0,1,2))
        
        nframe = data_in['properties']['nframe'];
        natm = len(data_in['elements']);
        norbs = data_in['properties']['norbs'];
        
        pos = data_in['pos'].reshape([-1,3]);
        batch = torch.tensor([int(i//natm) for i in range(len(pos))]).to(self.device);
        num_nodes = len(pos);
        edge_src, edge_dst = radius_graph(x=pos, r=self.max_radius, batch=batch);

        self_edge = torch.tensor([i for i in range(num_nodes)]).to(self.device);
        edge_src = torch.cat((edge_src, self_edge));
        edge_dst = torch.cat((edge_dst, self_edge));

        edge_vec = pos[edge_src] - pos[edge_dst];
        num_neighbors = len(edge_src) / num_nodes;

        sh = o3.spherical_harmonics(l = self.irreps_sh, 
                                    x = edge_vec, 
                                    normalize=True, 
                                    normalization='component').to(self.device)
        
        rnorm = edge_vec.norm(dim=1);
        crit1, crit2 = rnorm<self.max_radius, rnorm>self.min_radius;
        emb = (torch.cos(rnorm/self.max_radius*torch.pi)+1)/2; 
        emb = (emb*crit1*crit2 + (~crit2)).reshape(len(edge_src),1);

        f_in = torch.tensor([self.element_embedding[u] for u in data_in['elements']]*nframe,
                            dtype=torch.float).to(self.device);
        
        CC_ind = torch.argwhere(f_in[edge_src][:,0]*f_in[edge_dst][:,0]).reshape(-1);
        HH_ind = torch.argwhere(f_in[edge_src][:,1]*f_in[edge_dst][:,1]).reshape(-1);
        CH_ind = torch.argwhere(f_in[edge_src][:,0]*f_in[edge_dst][:,1]).reshape(-1);
        
        edge_feature = self.tp1(f_in[edge_src], sh, self.fc1(emb));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        edge_feature = self.tp2(node_feature[edge_src], sh, self.fc2(emb));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5);
        
        edge_HH = self.tpHH(node_feature[edge_src[HH_ind]], node_feature[edge_dst[HH_ind]], self.fcHH(emb[HH_ind]));
        edge_CC = self.tpCC(node_feature[edge_src[CC_ind]], node_feature[edge_dst[CC_ind]], self.fcCC(emb[CC_ind]));
        edge_CH = self.tpCH(node_feature[edge_src[CH_ind]], node_feature[edge_dst[CH_ind]], self.fcCH(emb[CH_ind]));
        edge_HH = self.transformer.transform(edge_HH, 2,2);
        edge_CC = self.transformer.transform(edge_CC, 3,3);
        edge_CH = self.transformer.transform(edge_CH, 3,2);
        
        Vmat = [torch.zeros([norbs, norbs], dtype=torch.float).to(self.device) for i in range(nframe)];
        
        map1 = [14+16*(ele=='C') for ele in data_in['elements']];
        map1 = [sum(map1[:i]) for i in range(len(map1)+1)];
        for i in range(len(HH_ind)):
            u1,u2 = edge_src[HH_ind[i]],edge_dst[HH_ind[i]];
            frame = batch[u1];
            v1 = u1-frame*natm;
            v2 = u2-frame*natm;
            Vmat[frame][map1[v1]:map1[v1+1], map1[v2]:map1[v2+1]] = edge_HH[i];
        for i in range(len(CC_ind)):
            u1,u2 = edge_src[CC_ind[i]],edge_dst[CC_ind[i]];
            frame = batch[u1];
            v1 = u1-frame*natm;
            v2 = u2-frame*natm;
            Vmat[frame][map1[v1]:map1[v1+1], map1[v2]:map1[v2+1]] = edge_CC[i];
        for i in range(len(CH_ind)):
            u1,u2 = edge_src[CH_ind[i]],edge_dst[CH_ind[i]];
            frame = batch[u1];
            v1 = u1-frame*natm;
            v2 = u2-frame*natm;
            Vmat[frame][map1[v1]:map1[v1+1], map1[v2]:map1[v2+1]] = edge_CH[i];
        
        Vmat = torch.stack([(V+V.T)/2 for V in Vmat]);
        
        return Vmat;


