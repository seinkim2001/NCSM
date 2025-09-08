import argparse
import random
import numpy as np
import networkx as nx
import pandas as pd

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_networkx
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn.conv import MessagePassing
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.datasets import Planetoid

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


# Set random seed
def set_seed(seed: int = 1) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_spd_matrix(G, S, max_spd: int = 5) -> np.ndarray:
    """Generating Shortest Path Distance Matrix with Maximum Threshold."""
    spd_matrix = np.zeros((G.number_of_nodes(), len(S)), dtype=np.float32)
    for i, node_S in enumerate(S):
        for node, length in nx.shortest_path_length(G, source=node_S).items():
            spd_matrix[node, i] = min(length, max_spd)
    return spd_matrix


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 5
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest AUC: {result[:, 3].max():.2f}')
            print(f'Highest AP: {result[:, 4].max():.2f}')
            print(f'Final Train: {result[argmax, 0]:.2f}')
            print(f'Final Test: {result[argmax, 2]:.2f}')
            print(f'Final AUC: {result[argmax, 3]:.2f}')
            print(f'Final AP: {result[argmax, 4]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                test_auc = r[r[:, 1].argmax(), 3].item()
                test_ap = r[r[:, 1].argmax(), 4].item()
                best_results.append((train1, valid, train2, test, test_auc, test_ap))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'Final Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'AUC Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 5]
            print(f'AP Test: {r.mean():.2f} ± {r.std():.2f}')


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the "Inductive Representation Learning on
    Large Graphs" paper."""

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr, emb_ea):
        edge_attr = torch.mm(edge_attr, emb_ea)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, edge_attr)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, edge_attr, x, emb_ea, adj_t, split_edge, optimizer, batch_size):
    edge_index = adj_t

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t, edge_attr, emb_ea)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def test(model, predictor, edge_attr, x, emb_ea, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t, edge_attr, emb_ea)

    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    total_preds = torch.cat((pos_test_pred, neg_test_pred), dim=0)
    labels = torch.cat((torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)), dim=0)
    auc = roc_auc_score(labels.cpu(), torch.round(total_preds.cpu()))
    ap_score = average_precision_score(labels.cpu(), torch.round(total_preds.cpu()))

    results = {}

    for K in [20]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        auc_results = {'AUC': auc}[f'AUC']
        ap_results = {'AP': ap_score}[f'AP']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits, auc_results, ap_results)
    return results


def compute_centrality(G, centrality_type):
    if centrality_type == 'DC':
        return nx.degree_centrality(G)
    elif centrality_type == 'EC':
        return nx.eigenvector_centrality(G, max_iter=1000)
    elif centrality_type == 'BC':
        return nx.betweenness_centrality(G)
    elif centrality_type == 'CC':
        return nx.closeness_centrality(G)
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")


def compute_similarity(G, u, v, similarity_type):
    if similarity_type == 'JA':
        return next(nx.jaccard_coefficient(G, [(u, v)]), (0, 0, 0))[2]
    elif similarity_type == 'AA':
        return next(nx.adamic_adar_index(G, [(u, v)]), (0, 0, 0))[2]
    elif similarity_type == 'RA':
        return next(nx.resource_allocation_index(G, [(u, v)]), (0, 0, 0))[2]
    elif similarity_type == 'PA':
        return next(nx.preferential_attachment(G, [(u, v)]), (0, 0, 0))[2]
    elif similarity_type == 'CN':
        return len(list(nx.common_neighbors(G, u, v)))
    elif similarity_type == 'Salton':
        cn = len(list(nx.common_neighbors(G, u, v)))
        degrees = nx.degree(G, [u, v])
        try:
            return cn / (degrees[u] * degrees[v])**0.5
        except ZeroDivisionError:
            return 0
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")


def get_hc_features(G, samples_edges, centrality_type, similarity_type):
    centralities = compute_centrality(G, centrality_type)
    feats = []
    for u, v in samples_edges:
        similarity = compute_similarity(G, u, v, similarity_type)
        u_centrality = centralities[u]
        feats.append([similarity, u_centrality])
    return feats


def load_dataset(name, device):
    split_edge = None

    if name in ['ogbl-ddi']:
        dataset = PygLinkPropPredDataset(name=name)
        split_edge = dataset.get_edge_split()
    elif name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
    elif name == 'CiteSeer':
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
    elif name == 'PubMed':
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
    else:
        raise ValueError("Dataset not supported")

    data = dataset[0]
    edge_index = data.edge_index.to(device)
    nx_graph = to_networkx(data, to_undirected=True)

    edgenp = edge_index.cpu().numpy()
    edges_reshaped = np.reshape(edgenp, (-1, 2), order='F')

    return data, nx_graph, edges_reshaped, split_edge


def main():
    parser = argparse.ArgumentParser(description='NCSM Link Prediction')
    parser.add_argument('--dataset', type=str, default='ogbl-ddi')
    parser.add_argument('--centrality', type=str, default='BC')
    parser.add_argument('--similarity', type=str, default='JA')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()

    set_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_steps = 1
    num_layers = 2
    hidden_channels = 256
    dropout = 0.3
    batch_size = 64 * 1024
    lr = 0.003
    eval_steps = 5
    num_samples = 1
    node_emb = 256

    data, nx_graph, edges_reshaped, split_edge = load_dataset(args.dataset, device)
    feat_train = np.array(get_hc_features(nx_graph, edges_reshaped, args.centrality, args.similarity))
    df_save = pd.DataFrame(feat_train, columns=['j_coefficient', 'u_centrality'])
    df_save.to_csv('jaccard_degree.csv', index=False)

    alpha = 0.5
    edge_attr = feat_train[:, 0] * alpha + feat_train[:, 1] * alpha
    edge_attr = np.reshape(edge_attr, (edge_attr.size, 1))
    edge_attr = torch.from_numpy(edge_attr).float().to(device)
    max_attr = torch.max(edge_attr)
    min_attr = torch.min(edge_attr)
    edge_attr = (edge_attr - min_attr) / (max_attr - min_attr + 1e-15)
    edge_index = data.edge_index.to(device)
    model = GraphSAGE(node_emb, hidden_channels, hidden_channels, num_layers, dropout).to(device)

    emb = torch.nn.Embedding(data.num_nodes, node_emb).to(device)
    emb_ea = torch.nn.Embedding(num_samples, node_emb).to(device)
    predictor = LinkPredictor(hidden_channels, hidden_channels, 1,
                              num_layers + 1, dropout).to(device)

    evaluator = Evaluator(name=args.dataset)
    loggers = {'Hits@20': Logger(args.runs)}

    for run in range(args.runs):
        random.seed(run)
        torch.manual_seed(run)
        torch.nn.init.xavier_uniform_(emb.weight)
        torch.nn.init.xavier_uniform_(emb_ea.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(emb_ea.parameters()) + list(predictor.parameters()), lr=lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, edge_attr, emb.weight, emb_ea.weight, edge_index, split_edge,
                          optimizer, batch_size)

            if epoch % eval_steps == 0:
                results = test(model, predictor, edge_attr, emb.weight, emb_ea.weight, edge_index, split_edge,
                                evaluator, batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits, auc, ap_score = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'AUC: {100 * auc:.2f}%, '
                              f'AP: {100 * ap_score:.2f}%, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == '__main__':
    main()
