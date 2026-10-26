import torch

# RelNet layers without the relation graph of tasks.build_relation_graph. With the DistMult message and
# sum aggregation, the sum over relation-graph edges (r_1, r_2, v) factors through the shared entities v:
#   y_v    = sum_{r_2 in S(v)} x[r_2]
#   m[r_1] = sum_{v : r_1 in D(v)} p_v * y_v  -  x[r_1] * sum_{v : r_1 in D(v)} p_v   (second term for hh / tt only)
# i.e. two sparse products by the entity x relation incidence matrices.

# flavour -> (source, destination) incidence; rspmm aggregates towards edge_index[0] = r_1
FLAVOURS = {"hh": ("H", "H"), "ht": ("T", "H"), "th": ("H", "T"), "tt": ("T", "T")}


def incidence(data):
    key = (int(data.num_edges), str(data.edge_index.device), data.edge_index.data_ptr())
    cached = getattr(data, "_relation_incidence", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    size = (int(data.num_nodes), int(data.num_relations))
    out = {}
    for side, entity in (("H", data.edge_index[0]), ("T", data.edge_index[1])):
        index = torch.stack([entity, data.edge_type]).unique(dim=1)
        A = torch.sparse_coo_tensor(index, torch.ones(index.size(1), device=index.device), size).coalesce()
        out[side] = (A, A.t().coalesce())
    data._relation_incidence = (key, out)
    return out


def conv(layer, input, boundary, relation, inc, flavour):
    if layer.message_func != "distmult" or layer.aggregate_func != "sum":
        raise ValueError("relation_graph_free requires message_func=distmult and aggregate_func=sum")
    src, dst = FLAVOURS[flavour]
    A_src, AT_dst = inc[src][0], inc[dst][1]
    batch_size, num_relations, dim = input.shape
    x = input.transpose(0, 1).reshape(num_relations, batch_size * dim)
    p = relation.transpose(0, 1).reshape(relation.shape[1], batch_size * dim)
    if src == dst:
        # the two terms nearly cancel on some relations, float32 loses precision there
        x64, p64, A_src, AT_dst = x.double(), p.double(), A_src.double(), AT_dst.double()
        m = torch.sparse.mm(AT_dst, p64 * torch.sparse.mm(A_src, x64)) - x64 * torch.sparse.mm(AT_dst, p64)
        m = m.to(x.dtype)
    else:
        m = torch.sparse.mm(AT_dst, p * torch.sparse.mm(A_src, x))
    m = m.reshape(num_relations, batch_size, dim).transpose(0, 1) + boundary
    return layer.update(m, input)


def relnet_layer(relnet, i, layer_input, boundary, node_representations, inc):
    hidden = 0
    for flavour in FLAVOURS:
        layer = getattr(relnet, "layers_" + flavour)[i]
        relation = layer.relation_projection(node_representations)
        hidden = hidden + conv(layer, layer_input, boundary, relation, inc, flavour)
    return hidden
