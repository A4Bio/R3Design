import torch
import torch.nn as nn
from modules import RDesignLayer
from .feature import RNAFeatures

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


class R3Design_Model(nn.Module):
    def __init__(self, args):
        super(R3Design_Model, self).__init__()

        self.device = 'cuda:0'
        self.smoothing = args.smoothing
        self.node_features = self.edge_features = args.hidden
        self.hidden_dim = args.hidden
        self.vocab = args.vocab_size
        self.n_iter = args.n_iter

        self.features = RNAFeatures(
            args.hidden, args.hidden, 
            top_k=args.k_neighbors, 
            dropout=args.dropout,
            node_feat_types=args.node_feat_types, 
            edge_feat_types=args.edge_feat_types,
            args=args
        )

        layer = RDesignLayer
        self.W_s = nn.Embedding(args.vocab_size, self.hidden_dim)
        self.encoder_layers = nn.ModuleList([
            layer(self.hidden_dim, self.hidden_dim*2, dropout=args.dropout)
            for _ in range(args.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([
            layer(self.hidden_dim, self.hidden_dim*2, dropout=args.dropout)
            for _ in range(args.num_decoder_layers)])

        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        )

        self.readout = nn.Linear(self.hidden_dim, args.vocab_size, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.base_pro = nn.Linear(args.vocab_size, self.hidden_dim)

    def forward(self, X, S, mask):
        X, S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask) 

        init_S_prob = torch.zeros((S.shape[0], self.vocab), device=S.device) + 1. / self.vocab
        S_prob = init_S_prob.clone()
        logits_lst = []
        for t in range(self.n_iter):
            # h_V = h_V + S_prob
            for enc_layer in self.encoder_layers:
                h_V, h_E = enc_layer(X, h_V, h_E, E_idx, batch_id)

            for dec_layer in self.decoder_layers:
                h_V, h_E = dec_layer(X, h_V, h_E, E_idx, batch_id)

            graph_embs = []
            for b_id in range(batch_id[-1].item()+1):
                b_data = h_V[batch_id == b_id].mean(0)
                graph_embs.append(b_data)
            graph_embs = torch.stack(graph_embs, dim=0)
            graph_prjs = self.projection_head(graph_embs)

            logits = self.readout(h_V) * S_prob
            S_prob = torch.softmax(logits, -1)
            logits_lst.append(logits)
        return logits_lst, S, graph_prjs

    def sample(self, X, S, mask=None):      
        X, gt_S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask) 

        init_S_prob = torch.zeros((S.shape[0], self.vocab), device=S.device) + 1. / self.vocab
        S_prob = init_S_prob.clone()

        for t in range(self.n_iter):
            for enc_layer in self.encoder_layers:
                h_V, h_E = enc_layer(X, h_V, h_E, E_idx, batch_id)

            for dec_layer in self.decoder_layers:
                h_V, h_E = dec_layer(X, h_V, h_E, E_idx, batch_id)

            logits = self.readout(h_V) * S_prob
            S_prob = torch.softmax(logits, -1)
        return logits, gt_S