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
# fix hyper-parameters
parser = argparse.ArgumentParser(description='EEG-CGS for Seizure Analysis using EEG Graphs')
parser.add_argument('--graph_type', type=str, default='distance')
parser.add_argument('--div_sz', type=int, default=2)
parser.add_argument('--div_nonsz', type=int, default=5)
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
test_loader = dataloaders['test'].dataset
total_samples = len(test_loader)

final_adj = []
final_features = []
final_labels = []
ano_label = []
seizure_label = []

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

print('#######################################')
print('SEIZURE DETECTION...')
print('Generating graphs...')

final_adj = []
final_features = []
final_labels = []
ano_label = []
seizure_label = []

with tqdm(total = total_samples, desc = 'EEG graph') as progressbar:
    for features, y, clip_len, adj_mat, _ in test_loader:
        adj = sp.csr_matrix(adj_mat)
        features = torch.reshape(features, (19, -1))
        nb_nodes = features.shape[0]
        features = torch.reshape(features, (nb_nodes, -1))
        adj = (adj + sp.eye(adj.shape[0])).todense()
        adj = torch.FloatTensor(adj)
        features = torch.FloatTensor(features)
        label = np.zeros((nb_nodes, 1), dtype=np.uint8)
        final_adj.append(adj.detach())
        final_features.append(features)
        # final_labels.append(labels)
        ano_label.append((torch.FloatTensor(label)))
        seizure_label.append(y)
        progressbar.update(1)

seizure_label = torch.cat(seizure_label, 0)
adj, features, ano_label, save_seizure_label = sz_set(final_adj, final_features, ano_label,
                                                      seizure_label, nb_nodes, args.div_sz, args.div_nonsz)
adj = sp.csr_matrix(adj)
dgl_graph = make_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
adj = (adj + sp.eye(adj.shape[0])).todense()
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
print('Computing anomaly scores for channels!', flush=True)
all_auc = []

with tqdm(total=args.auc_R_samp) as pbar_test:
    pbar_test.set_description('Testing...')
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
seizure_predicted = []
ch = 19
predicted = [ano_score_final[x:x+ch] for x in range(0, len(ano_score_final), ch)]
for id in range(0, len(predicted)):
    if (predicted[id]> args.tau1).sum() >= args.tau2:
        seizure_predict = 1
    else:
        seizure_predict = 0
    seizure_predicted.append(seizure_predict)

true = [int(np.mean(save_seizure_label[x:x+ch])) for x in range(0, len(ano_label), ch)]
scores_dict = dict_metrics(y_pred=seizure_predicted, y=true, y_prob=None, file_names=None, average="binary")
print(scores_dict)

result_table = pd.DataFrame(columns=['exp_id', 'graph_type', 'metrics'])
result_table = result_table.append({'exp_id': args.exp_id,
                                    'graph_type': args.graph_type,
                                    'metrics':scores_dict}, ignore_index=True)
result_table.to_csv('results/table2.csv', mode='a', index=True, header=False)

print('TESTING DONE!')
print('------------------------------')
print('Accuracy on Normal/Seizure clips: {:.4f}'.format(scores_dict['acc']))