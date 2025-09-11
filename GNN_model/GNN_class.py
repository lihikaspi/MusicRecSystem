from torch_geometric.nn import RGCNConv, HeteroConv
import torch.nn.functional as F
import torch.nn as nn

class WeightedRGCN(nn.Module):
    def __init__(self, x_dict, hidden_dim, out_dim, num_relations):
        super().__init__()
        # Optional: store input dims per node type
        self.node_types = list(x_dict.keys())
        self.convs = nn.ModuleList()
        
        # HeteroConv: handles multiple node types and relations
        self.conv1 = HeteroConv({
            ('user', 'interacts', 'item'): RGCNConv(x_dict['user'].size(1), hidden_dim, num_relations),
            ('item', 'rev_interacts', 'user'): RGCNConv(x_dict['item'].size(1), hidden_dim, num_relations)
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('user', 'interacts', 'item'): RGCNConv(hidden_dim, out_dim, num_relations),
            ('item', 'rev_interacts', 'user'): RGCNConv(hidden_dim, out_dim, num_relations)
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict, edge_type_dict):
        x_dict = self.conv1(x_dict, edge_index_dict, edge_type_dict)

        # optional: pass edges weights
        # x_dict = self.conv1(x_dict, edge_index_dict, edge_type_dict, edge_weight=edge_weight_dict)

        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict, edge_type_dict)
        return x_dict
