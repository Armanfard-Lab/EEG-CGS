from data_utils import *
from model_utils import *
from helper_functions import *
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import dgl
import argparse
from tqdm import tqdm
import scipy.sparse as sp
import random

sampled_dir = "data/data_tusz"
raw_dir = "data/data_tusz"
# The full dataset can be downloaded at: https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
fft_dir = "data/data_tusz/fft_small/small/clip_len60"

#############################################################
parser = argparse.ArgumentParser(description='EEG-CGS for Seizure Analysis using EEG Graphs')
parser.add_argument('--graph_type', type=str, default='distance')
parser.add_argument('--div_train', type=int, default=50) #1000, 500, 350, 260, 200
parser.add_argument('--fts_size', type=int, default=6000)
parser.add_argument('--embed_dim', type=int, default=256) #16, 32, 128, 256, 512
parser.add_argument('--subgraph_size', type=int, default=4) # 1, 2, 4, 6, 8
parser.add_argument('--batch_size', type=int, default=285)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--drop_prob', type=float, default=0)
parser.add_argument('--patience', type=int, default=400)
parser.add_argument('--lbda', type=float, default=0.6) # 0, 0.2, 0.4, 0.6, 0.8, 1
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=55) #100
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
print('TRAINING...')
print('Generating training graphs...')

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
dgl_graph = make_dgl_graph(adj)

# Total of nodes in graph
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = 2

adj = (adj + sp.eye(adj.shape[0])).todense()
adj = torch.FloatTensor(adj[np.newaxis]).to(args.device)
raw_fts = torch.FloatTensor(features[np.newaxis]).to(args.device)
fts = raw_fts # set for postive subgraphs
print('EEG graphs ready!')
print('------------------------------')


#############################################################

# Initialize model
model = EEG_CGS(ft_size, args.embed_dim, 'prelu', args.negsamp_ratio).to(args.device)
bce_con = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(args.device))
mse_rec = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch)

wait = 0
choose_best = 1e9
is_best_epoch = 0
batch = nb_nodes // args.batch_size + 1

# Training
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Start training...')
    for epoch in range(args.num_epoch):
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        loss_total = 0

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

            # finalize adj1 and adj2
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

            # pass through model
            logits, rec1pos, rec2pos = model(adj1, adj2, sub1neg_fts, sub2neg_fts, sub1pos_fts, sub2pos_fts)

            # compute losses
            #loss_con1 = bce_con(logits, samp_bat)
            loss_con = torch.mean(bce_con(logits, samp_bat))
            loss_rec = (mse_rec(rec1pos[:, -2, :], sub1pos_fts[:, -1, :]) + mse_rec(rec2pos[:, -2, :], sub2pos_fts[:, -1, :])) / 2
            loss = args.lbda * loss_con + (1 - args.lbda) * loss_rec

            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()

            if not is_fin_bat:
                loss_total += loss
        # print out loss_mean
        loss_mean = (loss_total * args.batch_size + loss * curr_bat_dim) / nb_nodes

        if loss_mean < choose_best:
            choose_best = loss_mean
            is_best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), 'checkpoints/exp_{}.pkl'.format(args.exp_id))
        else:
            wait += 1

        if wait == args.patience:
            print('Early stopping!', flush=True)
            break
        scheduler.step()
        pbar.set_postfix(loss = loss_mean, lr = optimizer.param_groups[0]['lr'])
        pbar.update(1)

print('TRAINING DONE!')

