import torch
from torch import nn

from . import tasks, layers
from trix.base_nbfnet import BaseNBFNet

class TRIX(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg, trix_cfg):
        super(TRIX, self).__init__()
        self.num_layer = trix_cfg.num_layer
        self.feature_dim = trix_cfg.feature_dim
        self.num_mlp_layer = trix_cfg.num_mlp_layer

        self.relation_model = nn.ModuleList()
        self.entity_model = nn.ModuleList()

        for i in range(self.num_layer):
            self.relation_model.append(RelNet(**rel_model_cfg))
            self.entity_model.append(EntityNet(**entity_model_cfg))

        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layer - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

        
    def forward(self, data, batch):
        # part 1 embedding initialization
        h_index_r = batch[:, 0, 2]
        h_index, t_index, r_index = batch.unbind(-1)
        rel_graph = data.relation_adj

        batch_size = len(h_index_r)
        relation_representations = torch.ones(batch_size, rel_graph["hh"].num_nodes, self.feature_dim, device=h_index_r.device)

        query = relation_representations[torch.arange(batch_size, device=r_index[:, 0].device), r_index[:, 0]]
        index_1 = h_index[:, 0].unsqueeze(-1).expand_as(query)
        index_2 = t_index[:, 0].unsqueeze(-1).expand_as(query)
        node_representations = torch.zeros(batch_size, data.num_nodes, self.feature_dim, device=h_index[:, 0].device)
        node_representations.scatter_add_(1, index_1.unsqueeze(1), query.unsqueeze(1))
        node_representations.scatter_add_(1, index_2.unsqueeze(1), query.unsqueeze(1) * (-1))

        # part 2 iterative embedding update: ( entity 1 - relation 1 ) * 3 rounds
        for i in range(self.num_layer):
            node_representations = self.entity_model[i](data, batch, node_representations, relation_representations)
            relation_representations = self.relation_model[i](data, batch, relation_representations, node_representations)

        # part 3 get score
        index = r_index.unsqueeze(-1).expand(-1, -1, relation_representations.shape[-1]) 
        feature = relation_representations.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score


class RelNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers_hh = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_hh.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )
            
        self.layers_ht = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_ht.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )
            
        self.layers_th = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_th.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )
            
        self.layers_tt = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_tt.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def forward(self, data, batch, relation_representations, node_representations):
        rel_graph = data.relation_adj
        h_index = batch[:, 0, 2]

        batch_size = len(h_index)
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        boundary = torch.ones(batch_size, rel_graph["hh"].num_nodes, self.dims[0], device=h_index.device)

        size = (rel_graph["hh"].num_nodes, rel_graph["hh"].num_nodes)
        edge_weight_hh = torch.ones(rel_graph["hh"].num_edges, device=h_index.device)
        edge_weight_ht = torch.ones(rel_graph["ht"].num_edges, device=h_index.device)
        edge_weight_th = torch.ones(rel_graph["th"].num_edges, device=h_index.device)
        edge_weight_tt = torch.ones(rel_graph["tt"].num_edges, device=h_index.device)

        hiddens = []
        layer_input = relation_representations

        for i in range(len(self.layers_hh)):
            self.layers_hh[i].relation = node_representations
            self.layers_ht[i].relation = node_representations
            self.layers_th[i].relation = node_representations
            self.layers_tt[i].relation = node_representations

            hidden_hh = self.layers_hh[i](layer_input, query, boundary, rel_graph["hh"].edge_index, rel_graph["hh"].edge_type, size, edge_weight_hh)
            hidden_ht = self.layers_ht[i](layer_input, query, boundary, rel_graph["ht"].edge_index, rel_graph["ht"].edge_type, size, edge_weight_ht)
            hidden_th = self.layers_th[i](layer_input, query, boundary, rel_graph["th"].edge_index, rel_graph["th"].edge_type, size, edge_weight_th)
            hidden_tt = self.layers_tt[i](layer_input, query, boundary, rel_graph["tt"].edge_index, rel_graph["tt"].edge_type, size, edge_weight_tt)

            hidden = hidden_hh + hidden_ht + hidden_th + hidden_tt
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)

            layer_input = hidden
            relation_representations = layer_input
            
        node_query = query.unsqueeze(1).expand(-1, rel_graph["hh"].num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]
     
        return output
    

class EntityNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, input_dim))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, t_index, r_index, node_representations, separate_grad=False):
        batch_size = len(h_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index_1 = h_index.unsqueeze(-1).expand_as(query)
        index_2 = t_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index_1.unsqueeze(1), query.unsqueeze(1))
        boundary.scatter_add_(1, index_2.unsqueeze(1), query.unsqueeze(1) * (-1))
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = node_representations

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, batch, node_representations, relation_representations):
        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = torch.ones_like(relation_representations).to(relation_representations.device)

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (t_index[:, [0]] == t_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], t_index[:, 0],  r_index[:, 0], node_representations)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]

        feature = self.mlp(feature)
        return feature