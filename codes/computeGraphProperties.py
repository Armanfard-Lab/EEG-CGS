from data_utils import use_dataloader
import torch
sampled_dir = "data/data_tusz"
raw_dir = "data/data_tusz"
fft_dir = "data/data_tusz/fft_small/clip_len60"
#adj_path = "data/data_tusz/fft_small/fixed_dist_adj/dist_adj.pkl"
dataloaders, datasets, scaler = use_dataloader(sampled_dir=fft_dir, raw_dir=raw_dir, train_batch_size=40,
                                               test_batch_size=128, time_step_size=1, clip_len=60, is_std = True,
                                               num_workers=8, adj_mat_dir = None, graph_type="correlation", top_edges = 3,
                                               sampl_ratio=1, seed=123, fft_dir=fft_dir)

train_loader = dataloaders['train'].dataset # only normal (nosz clips)
test_loader = dataloaders['test'].dataset # include sz and normal

# Find graphs
adj = []
for features, label, clip_len, adj_mat, _ in test_loader:
    adj.append(adj_mat)

print(adj[0])
print("Adjacency matrix shape is: " + str(adj[0].shape))
print("The number of graphs in test data is: " + str(len(adj)))

