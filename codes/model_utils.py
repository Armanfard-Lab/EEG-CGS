import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class GNN(nn.Module):

    def __init__(self, in_features, out_features, choose_actv, bias=True):
        super(GNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.choose_actv = nn.PReLU() if choose_actv == 'prelu' else choose_actv

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        #self.reset_parameters()

        for module in self.modules():
            self.reset_weights(module)

    def reset_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, clip, adj_mat, sparse=False):
        clip_fc = self.fc(clip)
        if sparse:
            graph_o = torch.unsqueeze(torch.spmm(adj_mat, torch.squeeze(clip_fc, 0)), 0)
        else:
            graph_o = torch.bmm(adj_mat, clip_fc)
        if self.bias is not None:
            graph_o += self.bias

        return self.choose_actv(graph_o)

class Avg(nn.Module):
    def __init__(self):
        super(Avg, self).__init__()

    def forward(self, emb):
        return torch.mean(emb, 1) # Take an average for a subgraph's embedding

class Sim(nn.Module):
    def __init__(self, v, negsampl_ratio):
        super(Sim, self).__init__()
        self.sim = nn.Bilinear(v, v, 1)

        for module in self.modules():
            self.reset_weights(module)

        self.negsampl_ratio = negsampl_ratio

    def reset_weights(self, module):
        if isinstance(module, nn.Bilinear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, v1, v2):
        all = []
        all.append(self.sim(v2, v1))
        sim_v1 = v1
        for _ in range(self.negsampl_ratio):
            sim_v1 = torch.cat((sim_v1[-1, :].unsqueeze(0), sim_v1[:-1, :]), dim=0)
            all.append(self.sim(v2, sim_v1))
        logits = torch.cat(tuple(all))
        return logits

class EEG_CGS(nn.Module):
    def __init__(self, input_features, embed, choose_actv, negsampl_ratio):
        super(EEG_CGS, self).__init__()
        self.gnn_e = GNN(input_features, embed, choose_actv)
        self.gnn_d = GNN(embed, input_features, choose_actv)
        self.avg = Avg()

        self.sim1 = Sim(embed, negsampl_ratio)
        self.sim2 = Sim(embed, negsampl_ratio)
        self.pair_dist = nn.PairwiseDistance(p=2)

    def forward(self,  adj1, adj2, sub1neg_fts, sub2neg_fts, sub1pos_fts, sub2pos_fts, sparse=False):
        # sub2neg_fts = sub2neg_fts
        emb1neg = self.gnn_e(sub1neg_fts, adj1, sparse)
        emb2neg = self.gnn_e(sub2neg_fts, adj2, sparse)
        emb1pos = self.gnn_e(sub1pos_fts, adj1, sparse)
        emb2pos = self.gnn_e(sub2pos_fts, adj2, sparse)

        # Contrastive module
        emb1 = emb1neg[:, -1, :]
        emb2 = emb2neg[:, -1, :]
        emb1_avg = self.avg(emb1neg[:, :-1, :])
        emb2_avg = self.avg(emb2neg[:, :-1, :])

        con1 = self.sim1(emb1_avg, emb2)
        con2 = self.sim2(emb2_avg, emb1)
        con = torch.cat((con1, con2), dim=-1).mean(dim=-1).unsqueeze(dim=-1)

        # Reconstruction module
        rec1pos = self.gnn_d(emb1pos, adj1, sparse)
        rec2pos = self.gnn_d(emb2pos, adj2, sparse)

        return con, rec1pos, rec2pos

    def test_phase(self, adj1, adj2, sub1neg_fts, sub2neg_fts, sub1pos_fts, sub2pos_fts, sparse=False):
        # sub2neg_fts = sub2neg_fts
        emb1neg = self.gnn_e(sub1neg_fts, adj1, sparse)
        emb2neg = self.gnn_e(sub2neg_fts, adj2, sparse)
        emb1pos = self.gnn_e(sub1pos_fts, adj1, sparse)
        emb2pos = self.gnn_e(sub2pos_fts, adj2, sparse)

        # Contrastive module
        emb1 = emb1neg[:, -1, :]
        emb2 = emb2neg[:, -1, :]
        emb1_avg = self.avg(emb1neg[:, :-1, :])
        emb2_avg = self.avg(emb2neg[:, :-1, :])

        con1 = self.sim1(emb1_avg, emb2)
        con2 = self.sim2(emb2_avg, emb1)
        con = torch.cat((con1, con2), dim=-1).mean(dim=-1).unsqueeze(dim=-1)

        # Reconstruction module
        rec1pos = self.gnn_d(emb1pos, adj1, sparse)
        rec2pos = self.gnn_d(emb2pos, adj2, sparse)
        rec1 = self.pair_dist(rec1pos[:, -2, :], sub1pos_fts[:, -1, :])
        rec2 = self.pair_dist(rec2pos[:, -2, :], sub2pos_fts[:, -1, :])
        rec = (rec1 + rec2) / 2

        return con, rec