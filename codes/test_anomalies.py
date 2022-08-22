from data_utils import *
from model_utils import *
from helper_functions import *
import sys
import torch
import dgl
import argparse
from tqdm import tqdm
import scipy.sparse as sp
import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sampled_dir = "data/data_tusz"
raw_dir = "data/data_tusz"
fft_dir = "data/data_tusz/fft_small/small/clip_len60"

#############################################################
parser = argparse.ArgumentParser(description='EEG-CGS for Seizure Analysis using EEG Graphs')
parser.add_argument('--graph_type', type=str, default='distance')
parser.add_argument('--div_train', type=int, default=50)
parser.add_argument('--fts_size', type=int, default=6000)
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=285)
parser.add_argument('--auc_R_samp', type=int, default=80)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--lbda', type=float, default=0.6)
parser.add_argument('--tau1', type=float, default=0.4)
parser.add_argument('--tau2', type=float, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--exp_id', type=str)
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

if args.graph_type == 'distance':
    adj_path = "data/data_tusz/fft_small/fixed_dist_adj/dist_adj.pkl"
else:
    adj_path = None

#############################################################
dataloaders, datasets, scaler = use_dataloader(sampled_dir=fft_dir, raw_dir=raw_dir, train_batch_size=40,
                                               test_batch_size=128, time_step_size=1, clip_len=60, is_std = True,
                                               num_workers=8, adj_mat_dir = None, graph_type="correlation", top_edges = 3,
                                               sampl_ratio=1, seed=123, fft_dir=fft_dir)

train_loader = dataloaders['train'].dataset
total_samples = len(train_loader)

# Random Seed
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
dgl.random.seed(args.seed)

#############################################################

print('#######################################')
print('TESTING Node Anomalies...')
print('Generating graphs...')

final_adj = []
final_features = []

with tqdm(total = total_samples, desc = 'EEG graph') as progressbar:
    for features, _, _ , adj_mat, _ in train_loader:
        adj = sp.csr_matrix(adj_mat)
        seq_length, nodes, fea = features.shape
        features = torch.reshape(features, (nodes, -1))
        nb_nodes = features.shape[0]
        adj = adj.todense()
        adj = torch.FloatTensor(adj)
        features = torch.FloatTensor(features)
        final_adj.append(adj.detach())
        final_features.append(features)
        progressbar.update(1)

final_adj = torch.cat(final_adj,0)
final_features = torch.cat(final_features,0)
print('Generating graphs DONE!')
print('------------------------------')

list_adj, features, len_adj = avg_clips(final_adj, final_features, nb_nodes, args.div_train)
idex = createIdx(int(len_adj/nb_nodes)-1)
adj = adj_concatenate(list_adj, len_adj, nb_nodes, idex)
adj = sp.csr_matrix(adj)
adj_ano, features_ano, ano_label, _, _ = add_anomaly(adj, features, mode = 'train')

# Total of nodes in graph
nb_nodes = features_ano.shape[0]
ft_size = features_ano.shape[1]
nb_classes = 2

dgl_graph = make_dgl_graph(adj_ano)
adj = (adj_ano + sp.eye(adj_ano.shape[0])).todense()
adj = torch.FloatTensor(adj[np.newaxis]).to(args.device)
raw_fts = torch.FloatTensor(features[np.newaxis]).to(args.device)
fts = raw_fts # set for postive subgraphs
print('EEG graphs ready!')
print('------------------------------')

print('#######################################')
print('TESTING...')

model = EEG_CGS(ft_size, args.embed_dim, 'prelu', args.negsamp_ratio).to(args.device)
model.load_state_dict(torch.load('checkpoints/exp_{}.pkl'.format(args.exp_id)))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
R_samp_score = np.zeros((args.auc_R_samp, nb_nodes))
batch = nb_nodes // args.batch_size + 1
print('Testing AUC on the channel-based anomalies!', flush=True)
all_auc = []

with tqdm(total=args.auc_R_samp) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_R_samp):
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        full_subneg = sampling_subgraph(dgl_graph, args.subgraph_size)
        full_subpos = sampling_subgraph(dgl_graph, args.subgraph_size)

        for bat_idx in range(batch):
            optimizer.zero_grad()
            is_fin_bat = (bat_idx == (batch - 1))

            if not is_fin_bat:
                idx = all_idx[bat_idx * args.batch_size: (bat_idx + 1) * args.batch_size]
            else:
                idx = all_idx[bat_idx * args.batch_size:]

            curr_bat_dim = len(idx)
            samp_bat = torch.unsqueeze(torch.cat((torch.ones(curr_bat_dim),
                                             torch.zeros(curr_bat_dim * args.negsamp_ratio))), 1).to(args.device)
            adj1 = []
            adj2 = []
            sub1neg_fts = []
            sub2neg_fts = []
            sub1pos_fts = []
            sub2pos_fts = []

            for i in idx:
                curr_adj1 = adj[:, full_subneg[i], :][:, :, full_subneg[i]]
                curr_fts1 = fts[:, full_subneg[i], :]
                curr_raw_fts1 = raw_fts[:, full_subneg[i], :]

                curr_adj2 = adj[:, full_subpos[i], :][:, :, full_subpos[i]]
                curr_fts2 = fts[:, full_subpos[i], :]
                curr_raw_fts2 = raw_fts[:, full_subpos[i], :]

                adj1.append(curr_adj1)
                adj2.append(curr_adj2)
                sub1neg_fts.append(curr_fts1)
                sub2neg_fts.append(curr_fts2)
                sub1pos_fts.append(curr_raw_fts1)
                sub2pos_fts.append(curr_raw_fts2)

            insert_adj_row = torch.zeros((curr_bat_dim, 1, args.subgraph_size)).to(args.device)
            insert_adj_col = torch.zeros((curr_bat_dim, args.subgraph_size + 1, 1)).to(args.device)
            insert_adj_col[:, -1, :] = 1.
            insert_fts_row = torch.zeros((curr_bat_dim, 1, ft_size)).to(args.device)

            # finalize adj1 and 2
            adj1 = torch.cat(adj1)
            adj1 = torch.cat((adj1, insert_adj_row), dim=1)
            adj1 = torch.cat((adj1, insert_adj_col), dim=2)
            adj2 = torch.cat(adj2)
            adj2 = torch.cat((adj2, insert_adj_row), dim=1)
            adj2 = torch.cat((adj2, insert_adj_col), dim=2)

            # finalize relevant subgraphs
            sub1neg_fts = torch.cat(sub1neg_fts)
            sub1neg_fts = torch.cat((sub1neg_fts[:, :-1, :], insert_fts_row, sub1neg_fts[:, -1:, :]), dim=1)
            sub2neg_fts = torch.cat(sub2neg_fts)
            sub2neg_fts = torch.cat((sub2neg_fts[:, :-1, :], insert_fts_row, sub2neg_fts[:, -1:, :]), dim=1)

            sub1pos_fts = torch.cat(sub1pos_fts)
            sub1pos_fts = torch.cat((sub1pos_fts[:, :-1, :], insert_fts_row, sub1pos_fts[:, -1:, :]), dim=1)
            sub2pos_fts = torch.cat(sub2pos_fts)
            sub2pos_fts = torch.cat((sub2pos_fts[:, :-1, :], insert_fts_row, sub2pos_fts[:, -1:, :]), dim=1)

            with torch.no_grad():
                logits_con, rec = model.test_phase(adj1, adj2, sub1neg_fts, sub2neg_fts, sub1pos_fts, sub2pos_fts)
                logits_con = torch.sigmoid(torch.squeeze(logits_con))

            if args.lbda == 1:
                if args.negsamp_ratio == 1:
                    score_con = - (logits_con[:curr_bat_dim] - logits_con[curr_bat_dim:]).cpu().numpy()
                else:
                    neg_score_con = logits_con[curr_bat_dim:].view(-1, curr_bat_dim).mean(dim=0)
                    pos_score_con = logits_con[:curr_bat_dim]

                    final_score = (neg_score_con - pos_score_con).cpu().numpy()
            elif args.lbda == 0:
                final_score = rec.cpu().numpy()
            else:
                delta1 = MinMaxScaler()
                delta2 = MinMaxScaler()
                if args.negsamp_ratio == 1:
                    score_con = - (logits_con[:curr_bat_dim] - logits_con[curr_bat_dim:]).cpu().numpy()
                else:
                    neg_score_con = logits_con[curr_bat_dim:].view(-1, curr_bat_dim).mean(dim=0)
                    pos_score_con = logits_con[:curr_bat_dim]
                    score_con = (neg_score_con - pos_score_con).cpu().numpy()
                score_con = delta1.fit_transform(score_con.reshape(-1, 1)).reshape(-1)
                score_rec = rec.cpu().numpy()
                score_rec = delta2.fit_transform(score_rec.reshape(-1, 1)).reshape(-1)
                final_score = args.lbda * score_con + (1 - args.lbda) * score_rec

            R_samp_score[round, idx] = final_score
        pbar_test.update(1)

ano_score_final = np.mean(R_samp_score, axis=0)
auc = roc_auc_score(ano_label, ano_score_final)
all_auc.append(auc)
print('AUC on nodes:{:.4f}'.format(auc), flush=True)

np.set_printoptions(threshold=sys.maxsize)

fpr, tpr, th = roc_curve(ano_label, ano_score_final)
result_table = pd.DataFrame(columns=['exp_id', 'graph_type', 'fpr','tpr','auc'])
result_table = result_table.append({'exp_id': args.exp_id,
                                    'graph_type': args.graph_type,
                                    'fpr':fpr,
                                    'tpr':tpr,
                                    'auc':auc}, ignore_index=True)
result_table.to_csv('results/table1.csv', mode='a', index=True, header=False)
best_thresh = threshold_max_f1(y_true=ano_label, y_prob=ano_score_final)
y_pred_all = (ano_score_final > best_thresh).astype(int)  # (batch_size, )
scores_dict = dict_metrics(y_pred=y_pred_all, y=ano_label[:,0], y_prob=ano_score_final, file_names=None, average="binary")

print(scores_dict)