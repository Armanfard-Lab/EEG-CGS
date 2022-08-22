"""
Some data preprocessing codes for TUSZ are adapted from https://github.com/tsy935/eeg-gnn-ssl
"""

from select_channels import *

from pathlib import Path
import pickle
import sklearn

import warnings
warnings.filterwarnings("ignore")
import os
from select_channels import *
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import pyedflib
import h5py
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import resample, correlate
from scipy.spatial.distance import euclidean
import scipy.sparse as sp
import networkx as nx
import dgl
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from collections import defaultdict

###########################################
# I - READ DATA
def getEDFfile(edf_file):
    """
    Input:
        edf_file (original file)
    Returns:
        signals: num_ch x num_datapoint
    """
    file = edf_file.signals_in_file
    samples = edf_file.getNSamples()[0]
    signals = np.zeros((file,samples))
    for i in range(file):
        try:
            signals[i,:] = edf_file.readSignal(i)
        except:
            pass
    return signals

def getSeizureEvents(edf_file):
    """
    Input:
        edf_file_name (original file)
    Returns:
        seizure_event: list of seizure events in seconds
    """
    tse_file = edf_file.split('.edf')[0] + '.tse_bi'
    seizure_event = []
    with open(tse_file) as f:
        for line in f.readlines():
            if "seiz" in line:  # check if seizure appears
                seizure_event.append([float(line.strip().split(' ')[0]), float(line.strip().split(' ')[1])])
    return seizure_event


def getChannelpositions(edf_file, verbose, labels, channels):
    """
    Input:
        edf_file (original file)
    Returns:
        channel_positions: list of channel positions
    """
    labels = list(labels)

    for i in range(len(labels)):
        labels[i] = labels[i].split('-')[0]

    channel_positions = []

    for ch in channels:
        try:
            channel_positions.append(labels.index(ch))
        except:
            if (verbose):
                print(edf_file + "can not find the channel " + ch)
            raise Exception("Channel(s) do not match in the EDF!")
    return channel_positions

###########################################
# II - PREPROCESSING ON EEG SIGNALS
#1. Swap channels
def swap_channel_pairs(channels):
    """
    Inputs:
        channels
    Outputs:
        list of tuples where each pair of channels are swapped
    """
    swap_pairs = []
    if ('EEG FP1' in channels) and ('EEG FP2' in channels):
        swap_pairs.append((channels.index('EEG FP1'), channels.index('EEG FP2')))
    if ('EEG Fp1' in channels) and ('EEG Fp2' in channels):
        swap_pairs.append((channels.index('EEG Fp1'), channels.index('EEG Fp2')))
    if ('EEG F3' in channels) and ('EEG F4' in channels):
        swap_pairs.append((channels.index('EEG F3'), channels.index('EEG F4')))
    if ('EEG F7' in channels) and ('EEG F8' in channels):
        swap_pairs.append((channels.index('EEG F7'), channels.index('EEG F8')))
    if ('EEG C3' in channels) and ('EEG C4' in channels):
        swap_pairs.append((channels.index('EEG C3'), channels.index('EEG C4')))
    if ('EEG T3' in channels) and ('EEG T4' in channels):
        swap_pairs.append((channels.index('EEG T3'), channels.index('EEG T4')))
    if ('EEG T5' in channels) and ('EEG T6' in channels):
        swap_pairs.append((channels.index('EEG T5'), channels.index('EEG T6')))
    if ('EEG O1' in channels) and ('EEG O2' in channels):
        swap_pairs.append((channels.index('EEG O1'), channels.index('EEG O2')))

    return swap_pairs

# 2. Resample all EEG signals with a fixed frequency of 200 Hz with n channels (n = 19) to h5 files
def resampled_sig(signals, freq_samp = 200, window_size = 4):
    """
    Input:
        signals: extracted from EDF files
    Returns:
        resampled
    """
    num_points = int(freq_samp * window_size)
    resampled = resample(signals, num = num_points, axis=1)

    return resampled

def resample_signal(edf_dir, save_dir):
    """
    Input:
        edf_dir: original EDF directory
        save_dir: save preprocessed H5 files to new directory
    Returns:
        Resampled signals in H5 files save to new directory
    """
    # Append all edf files
    print("Start Resampling Signals...!!")

    edf_files = []
    for path, subdir, file in os.walk(edf_dir):
        for name in file:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    # Check if any unusable files
    removed_files = []
    for i in tqdm(range(len(edf_files))):
        edf_file = edf_files[i]
        save_file = os.path.join(save_dir, edf_file.split('/')[-1].split('.edf')[0]+'.h5')

        #if not os.path.exists(save_file):
        #    os.mkdir(save_file)
        #print(save_file)

        if os.path.exists(save_file):
            continue
        try:
            f = pyedflib.EdfReader(edf_file)
        except BaseException:
            removed_files.append(edf_file)

        channel_positions = getChannelpositions(edf_file, False, f.getSignalLabels(), channels)
        signals = getEDFfile(f)
        signal_array = np.array(signals[channel_positions, :])
        freq_samp = f.getSampleFrequency(0)

        if freq_samp != freq:
            signal_array = resampled_sig(signal_array, freq = freq, window_size = int(signal_array.shape[1])/freq_samp)

        with h5py.File(save_file , 'w') as hf:
            hf.create_dataset('resampled_signal', data= signal_array)
            hf.create_dataset('resample_freq', data = freq)

    print("Resampling EEG signals DONE! There are {} failed files.".format(len(removed_files)))

# 3. Standardization the signals
class Standardization:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def scale(self, data_signal):
        standardize = (data_signal - self.mu)/self.sigma
        return standardize

    def inverse_scale(self, data_signal, tensor = False, is_device = None, masked = None):
        """
        Masked inverted scaling
        masked = (batch_size,) where channels are masked
        """
        mu = self.mu.copy()
        sigma = self.std.copy()

        if len(mu.shape) == 0:
            mu = [mu]
            sigma = [sigma]

        if tensor:
            # convert to torch tensor
            mu = torch.FloatTensor(mu)
            sigma = torch.FloatTensor(sigma)

            if is_device is not None:
                mu = mu.to(is_device)
                sigma = sigma.ti(is_device)

        inversed = data_signal * sigma + mu
        return inversed

def computeFFT(signals, size):
    """
    Inputs:
        signals = num_ch x num_datapoint
        size: length of positive frequency of FFT
    Outputs:
        log_fft: log amplitude of FFT (num_ch x num_datapoint)
        phase_fft: phase spectrum of FFT num_ch x num_datapoint
    """
    # Compute FFT
    fft_sig = fft(signals, n = size, axis = -1)

    # Remove negative frequency
    idx = int(np.floor(size/2))
    fft_sig = fft_sig[:,:idx]

    ampl = np.abs(fft_sig)
    ampl[ampl == 0] = 1e-8 # avoid log of 0

    log_fft = np.log(ampl)
    phase_fft = np.angle(fft_sig)

    return log_fft, phase_fft

def computeCorrelation (x_i, x_j, mode = 'valid', is_normalized = True):
    """
    Inputs:
        x_i, x_j: 2 feature vectors of 2 channels
    Outputs:
        xcorr: normalized corss-correlation of 2 feature vectors
    """
    xcorr = correlate(x_i, x_j, mode = mode)
    # Set normalized values for each feature vector
    norm1 = np.sum(np.absolute(x_i)**2)
    norm2 = np.sum(np.absolute(x_j)**2)
    if is_normalized and (norm1 != 0) and (norm2 != 0):
        scale = (norm1 * norm2) ** 0.5
        xcorr /= scale
    return xcorr

###########################################
# III - USE DATALOADERS
#1. Create EEG clips from H5 files
data_dir = 'data/small_file_markers/'

def makeEEGClips(h5_file, edf_file, clip_idex, time_step = 1, clip_len=60):
    """
    Convert entire EEG signals to clips
    Inputs:
        h5_file: resampled folder
        edf_file: original folder
        clip_idex: index at current clip
        time_step: length of each time step (s)
        clip_len:  slip size
    Outputs:
        all_clips: append all EEG clips (clip_len*freq, num_nodes/channels, time_step*freq)
        all_label: append all labels for each clip, 0 for normal and 1 for seizure
    """
    # Read H5 file from H5 created folder
    with h5py.File(h5_file, 'r') as h5:
        signal_array = h5["resampled_signal"][()]
        resampled_freq = h5["resample_freq"][()]
    assert resampled_freq == freq

    # Get seizure events
    seizure_events = getSeizureEvents(edf_file.split('.edf')[0])

    # Get length with frequency
    freq_slide_len = int(freq*clip_len)
    freq_time_step = int(freq*time_step)

    # Iterate through signals
    start_clip = clip_idex * freq_slide_len
    end_clip = start_clip + freq_slide_len
    current_slice = signal_array[:, start_clip:end_clip]

    start_time_step = 0
    total_time_steps = []

    while start_time_step <= current_slice.shape[1] - freq_time_step:
        end_time_step = start_time_step + freq_time_step
        current_time_step = current_slice[:, start_time_step:end_time_step]
        current_time_step, _ = computeFFT(current_time_step, size = freq_time_step)
        total_time_steps.append(current_time_step)
        start_time_step = end_time_step
    data_clip = np.stack(total_time_steps, axis = 0)

    # Check if seizure events appear in the current clip
    is_sz = 0
    for event in seizure_events:
        start_event = int(event[0] * freq)
        end_event = int(event[1] * freq)
        if not ((end_clip < start_event) or (start_clip > end_event)):
            is_sz = 1
            break
    return data_clip, is_sz

#2. Read Marker Files
def readMarkerfiles(split_type, sz_file, nonsz_file, cv_seed=42):
    """
    Convert entire EEG signals to clips
    Inputs:
        split_type: either 'train' or 'test'
        sz_file
        nonsz_file
    Outputs:
        combined_tuples: including both sz and nonsz for test
    """
    np.random.seed(cv_seed)
    sz_str = []
    nonsz_str = []

    sz_read = open(sz_file, "r")
    sz_str.extend(sz_read.readlines()) # ~ 3594 sz_file
    nonsz_read = open(nonsz_file, "r")
    nonsz_str.extend(nonsz_read.readlines()) # ~ 35,019 nonsz_file

    if split_type == 'train':
        combined_str = nonsz_str
        np.random.shuffle(combined_str)
        print('The total number of EEG training files is : ' + str(len(combined_str)))
    else:
        num_datapoints = len(sz_str)
        print('The number of seizure files in test is: ', num_datapoints)
        combined = sz_str + nonsz_str # only sz_str for localization
        #np.random.shuffle(combined)
        combined_str = combined
        print('The total number of EEG test files is : ' + str(len(combined_str)))

    combined_tuples = []
    for i in range(len(combined_str)):
        tuple = combined_str[i].strip("\n").split(",")
        tuple[1] = int(tuple[1])
        combined_tuples.append(tuple)

    #print_str = 'Number of EEG files in ' + \
     #   split_type + ': ' + str(len(combined_tuples))
    #print(print_str)

    return combined_tuples

# 3. Use dataloaders

class formulateData(Dataset):
    def __init__(self, sampled_dir, raw_dir, time_step_size=1, clip_len=60, is_std=True, scaler=None,
                 split='train', adj_mat_dir=None, graph_type=None, top_edges=None, sampl_ratio=1, seed=42, fft_dir=None):
        self.sampled_dir = sampled_dir # H5 dir
        self.raw_dir = raw_dir # edf dir
        self.time_step_size = time_step_size # int, (s)
        self.clip_len = clip_len # clip size (s)
        self.is_std = is_std # z-normalize in the training set
        self.scaler = scaler # for std
        self.split = split # train or test
        self.adj_mat_dir = adj_mat_dir # pre-computed adjacent matrix only available for Dist graphs (from Tang et al. 2022)
        self.graph_type = graph_type # Dist/Rand/Corr/DTF
        self.top_edges = top_edges # int, Corr/DTF
        self.sampl_ratio = sampl_ratio # ratio of positive to negative samples
        self.fft_dir = fft_dir # fft dir
        self.edf_files = []

        for path, subdir, file in os.walk(raw_dir):
            for name in file:
                if ".edf" in name:
                    self.edf_files.append(os.path.join(path,name))

        sz_file = os.path.join (data_dir, split + '_small_' + str(clip_len) + 's_sz.txt')
        nonsz_file = os.path.join (data_dir,split + '_small_' + str(clip_len) + 's_nosz.txt')
        #sz_file = os.path.join (data_dir, split + '_small_' + str(clip_len) + 's_sz_anno.txt')
        #nonsz_file = os.path.join (data_dir,split + '_small_' + str(clip_len) + 's_nosz.txt')
        self.file_tuples = readMarkerfiles(split, sz_file, nonsz_file, cv_seed=seed)
        self.size_tuples = len(self.file_tuples)

        # Electrode IDs based on 10-20 EEG system from TUSZ
        self.electrode_IDs = [electrode.split(' ')[-1] for electrode in channels]

        channel_targets = []

        for idx in range(len(self.file_tuples)):
            if self.file_tuples[idx][-1] == 0:
                channel_targets.append(0)
            else:
                channel_targets.append(1)
        self.channel_targets = channel_targets

    def __len__(self):
        return self.size_tuples

    def channel_targets(self):
        return self.channel_targets

    def random_swap(self, eeg_sig):

        swap = swap_channel_pairs(channels)
        eeg_sig_re = eeg_sig.copy()

        if(np.random.choice([True, False])):
            for pair in swap:
                eeg_sig_re[:, [pair[0], pair[1]],:] = eeg_sig[:, [pair[1], pair[0]], :]
        else:
            swap_pairs = None
        return eeg_sig_re, swap


    def compute_distance_graph(self, swap_channels = None):
        """
        Available pre-computed adj_mat Dist (Tang et al., 2022)
        """
        with open(self.adj_mat_dir,'rb') as pf:
            adj = pickle.load(pf)
            adj = adj[-1]
        adj_dist = adj.copy()

        if swap_channels is not None:
            for pair_ch in swap_channels:
                for i in range(adj.shape[0]):
                    adj_dist[pair_ch[0], i] = adj_dist[pair_ch[1], i]
                    adj_dist[pair_ch[1], i] = adj_dist[pair_ch[0], i]
                    adj_dist[i, pair_ch[0]] = adj_dist[i, pair_ch[1]]
                    adj_dist[i, pair_ch[1]] = adj_dist[i, pair_ch[0]]
                    adj_dist[i,i] = 1
                adj_dist[pair_ch[0], pair_ch[1]] = adj[pair_ch[1], pair_ch[0]]
                adj_dist[pair_ch[1], pair_ch[0]] = adj[pair_ch[0], pair_ch[1]]
        return adj_dist

    def compute_random_graph(self, clip, swap_ch = None):
        """
        Convert entire EEG signals to clips
        Inputs:
            clip: clip_len x num_ch x num_ft
            swap_ch: list of swapped indexes
        Outputs:
            adj: num_ch x num_ch
        """

        num_electrode = len(self.electrode_IDs)  # 19
        adj_rand = np.eye(num_electrode, num_electrode, dtype=np.float32) #diagonal is 1
        clip = np.transpose(clip, (1, 0, 2))  # (num_ch, clip_len, num_ft)
        assert clip.shape[0] == num_electrode

        clip = clip.reshape(num_electrode, -1)  # (num_ch, clip_len*num_ft)
        # print(clip.shape)

        for i in range(0, num_electrode):
            for j in range(i + 1, num_electrode):
                adj_rand[i, j] = 0.5
                adj_rand[j, i] = 0.5
        #adj_rand = abs(adj_rand)

        return adj_rand

    def compute_correlation_graph(self, clip, swap_ch = None):
       """
       Convert entire EEG signals to clips
        Inputs:
            clip: clip_len x num_ch x num_ft
            swap_ch: list of swapped indexes
        Outputs:
            adj: num_ch x num_ch
       """
       num_electrode = len(self.electrode_IDs) #19
       adj_corr = np.eye(num_electrode, num_electrode, dtype = np.float32) #diagonal is 1
       clip = np.transpose(clip, (1,0,2)) # (num_ch, clip_len, num_ft)
       assert clip.shape[0] == num_electrode

       clip = clip.reshape(num_electrode, -1) # (num_ch, clip_len*num_ft)

       electrode_IDs_dict = {}
       for i, electrode_ID in enumerate(self.electrode_IDs):
           electrode_IDs_dict[electrode_ID] = i
       if swap_ch is not None:
           for pair_nodes in swap_ch:
               node_i = [key for key, value in electrode_IDs_dict.items() if value == pair_nodes[0]][0]
               node_j = [key for key, value in electrode_IDs_dict.items() if value == pair_nodes[1]][0]
               electrode_IDs_dict[node_i] = pair_nodes[1]
               electrode_IDs_dict[node_j] = pair_nodes[0]
       #print(electrode_IDs_dict)

       for i in range(0, num_electrode):
           for j in range(i+1, num_electrode):
               corr = computeCorrelation(clip[i,:], clip[j,:], mode='valid', is_normalized = True)

               adj_corr[i,j] = corr
               adj_corr[j,i] = corr
       adj_corr = abs(adj_corr)

       if(self.top_edges is not None):
           adj_corr = keep_top_edges(adj_corr, top_edge = self.top_edges, directed = True)
       else:
           raise ValueError('Invalid value for top-edges!')
       return adj_corr

    def compute_dtf_graph(self, clip, swap_ch = None):
       """
       Convert entire EEG signals to clips
        Inputs:
            clip: clip_len x num_ch x num_ft
            swap_ch: list of swapped indexes
        Outputs:
            adj: num_ch x num_ch
       """
       num_electrode = len(self.electrode_IDs) #19
       adj_dtf = np.eye(num_electrode, num_electrode, dtype = np.float32) #diagonal is 1
       clip = np.transpose(clip, (1,0,2)) # (num_ch, clip_len, num_ft)
       assert clip.shape[0] == num_electrode

       clip = clip.reshape(num_electrode, -1) # (num_ch, clip_len*num_ft)

       electrode_IDs_dict = {}
       for i, electrode_ID in enumerate(self.electrode_IDs):
           electrode_IDs_dict[electrode_ID] = i
       if swap_ch is not None:
           for pair_nodes in swap_ch:
               node_i = [key for key, value in electrode_IDs_dict.items() if value == pair_nodes[0]][0]
               node_j = [key for key, value in electrode_IDs_dict.items() if value == pair_nodes[1]][0]
               electrode_IDs_dict[node_i] = pair_nodes[1]
               electrode_IDs_dict[node_j] = pair_nodes[0]
       #print(electrode_IDs_dict)
       all_corr = 0
       for i in range(0, num_electrode):
           for j in range(i+1, num_electrode):
               #if j != i:
               corr = computeCorrelation(clip[i, :], clip[j, :], mode='valid', is_normalized=True)
               for z in range(0, num_electrode):
                   if z != i and z!= j:
                       corr_re = computeCorrelation(clip[i, :], clip[z, :], mode='valid', is_normalized=True)
                       all_corr += corr_re 
                       denum_dtf = all_corr ** 2
                       dtf = corr / np.sqrt(denum_dtf)
                       adj_dtf[i,j] = dtf
       #adj_dtf = abs(adj_dtf)
       adj_dtf = sklearn.preprocessing.minmax_scale(adj_dtf, feature_range=(0, 1), axis=0, copy=True)

       if(self.top_edges is not None):
           adj_dtf = keep_top_edges(adj_dtf, top_edge= self.top_edges, directed = True)
       else:
           raise ValueError('Invalid value for top-edges!')
       return adj_dtf

    def __getitem__(self, idex):
        """
        Inputs:
            index: int
        Outputs:
            A tuple of (ft, label, clip_len, adj, writeout_file)
        """
        h5_file, sz_label = self.file_tuples[idex]
        clip_idx = int(h5_file.split('_')[-1].split('.h5')[0])
        edf_file = [file for file in self.edf_files if h5_file.split('.edf')[0] + '.edf' in file]
        edf_file = edf_file[0]

        # Check if fft dir is available
        if self.fft_dir is None:
            resampled_dir = os.path.join(self.sampled_dir, h5_file.split('.edf')[0] + '.h5')

            clip, is_sz = makeEEGClips(h5_file = resampled_dir, edf_file = edf_file, clip_idex= clip_idx,
                                                  time_step = self.time_step_size, clip_len = self.clip_len)
        else:
            with h5py.File(os.path.join(self.fft_dir, h5_file), 'r') as hf:
                clip = hf['slide'][()] #clip for localization

        swap_ch = None
        current_feature = clip.copy()

        # Standardize the data
        if self.is_std:
            current_feature = self.scaler.scale(current_feature)

        # Convert to tensors
        feature = torch.FloatTensor(current_feature)
        label = torch.FloatTensor([sz_label])
        clip_len = torch.LongTensor([self.clip_len])
        writeout_file = h5_file.split('.h5')[0]

        # Compute adjacency matrix for each graph type
        if self.graph_type == 'correlation':
            adj = self.compute_correlation_graph(clip, swap_ch)
        elif self.graph_type == 'random':
            adj = self.compute_random_graph(clip, swap_ch)
        elif self.graph_type == 'dtf':
            adj = self.compute_dtf_graph(clip, swap_ch)
        elif self.adj_mat_dir is not None:
            adj = self.compute_distance_graph(swap_ch)
        else:
            adj = []
        return (feature, label, clip_len, adj, writeout_file)

def use_dataloader (sampled_dir, raw_dir, train_batch_size, test_batch_size = None,
                    time_step_size=1, clip_len = 60, is_std = True, num_workers = 8,
                    adj_mat_dir = None, graph_type = None, top_edges = None, sampl_ratio = 1,
                    seed = 42, fft_dir = None):
    if (graph_type is not None) and (graph_type not in ['distance', 'random', 'correlation', 'dtf']):
        raise NotImplementedError

    if is_std:
        mean_dir = os.path.join(data_dir, 'mean_fft_'+str(clip_len)+'s.pkl')
        std_dir = os.path.join(data_dir, 'std_fft_'+str(clip_len)+'s.pkl')
        with open(mean_dir, 'rb') as f:
            mean = pickle.load(f)
        with open(std_dir, 'rb') as f:
            std = pickle.load(f)

        scaler = Standardization(mu=mean, sigma=std)
    else:
        scaler = None

    dataloaders = {}
    datasets = {}
    for split in ['train', 'test']:
        dataset = formulateData(sampled_dir = sampled_dir, raw_dir = raw_dir,
                                time_step_size = time_step_size, clip_len= clip_len,
                                is_std = is_std, scaler = scaler, split = split,
                                adj_mat_dir = adj_mat_dir, graph_type = graph_type,
                                top_edges= top_edges, sampl_ratio = sampl_ratio,
                                seed = seed, fft_dir = fft_dir)
        if split == 'train':
            shuffle = True
            batch_size = train_batch_size
        else:
            shuffle = False
            batch_size = test_batch_size
        loader = DataLoader(dataset = dataset, shuffle = shuffle,
                            batch_size = batch_size, num_workers = num_workers)
        dataloaders[split] = loader
        datasets[split] = dataset

    return dataloaders, datasets, scaler
###########################################
# IV - PROCESSING ON GRAPHS
#1. Avoid overly sconnected graphs
def keep_top_edges(adj_mat, top_edge = 3, directed = True):
    """
    Inputs:
        adj_mat: num_ch x num_ch
        top_edge: keep top strongest edges for each channel
        directed: only for Corr and DTF
    Outputs:
        adj_mat_new: num_ch x num_ch (with top edges)
    """
    adj_mat_no_edge = adj_mat.copy()

    for i in range(adj_mat_no_edge.shape[0]):
        adj_mat_no_edge[i, i] = 0
    top_edge_idx = (-adj_mat_no_edge).argsort(axis=-1)[:, :top_edge]
    masked_edge = np.eye(adj_mat.shape[0], dtype = bool)

    for i in range(0, top_edge_idx.shape[0]):
        for j in range(0, top_edge_idx.shape[1]):
            masked_edge[i, top_edge_idx[i,j]] = 1
            if not directed:
                masked_edge[top_edge_idx[i,j], i] = 1
    adj_mat_new = masked_edge * adj_mat
    return adj_mat_new

#2. Building DLG graphs
def adj_to_dlg_graph(adj_mat):
    """
    Convert adjacency matrix to dgl format
    Inputs:
        adj_mat: num_ch x num_ch
    Outputs:
        dgl graph
    """
    sparse_graph = nx.from_scipy_sparse_matrix(adj_mat)
    dgl_graph = dgl.DGLGraph(sparse_graph)

    return dgl_graph

#3. Add anomalies to graphs
def add_anomaly(adj_mat, features, mode = 'train'):
    """
    Corrupt both structural and contextual information
    Inputs:
        adj_mat:  nxn
        features: nxd
    Outputs:
        adj_mat (new): nxn
        features (new): nxd
        label: label of all nodes (normal/abnormal)
        copy_fea: labels of nodes being copied (structual)
        anomaly_idx: save index of node anomalies
    """
    if mode == 'train':
        # Set the percentage of anomalies in a total of nodes among all EEG clips
        p = 5
        q = 5
        ori_num_edge = np.sum(adj_mat)
        num_node = adj_mat.shape[0]
        label = np.zeros((num_node, 1), dtype=np.uint8)
        all_idx = list(range(num_node))
        random.shuffle(all_idx)
        anomaly_idx = all_idx[:p * q * 2]
        label[anomaly_idx, 0] = 1

        # Disturb adj_mat
        for q_ in range(q):
            current_nodes = anomaly_idx[q_ * p:(q_ + 1) * p]
            #print(current_nodes)
            for i in current_nodes:
                for j in current_nodes:
                    adj_mat[i, j] = 1.

            adj_mat[current_nodes, current_nodes] = 0.
            num_add_edge = np.sum(adj_mat) - ori_num_edge

        # Disturb features
        cluster = 50
        copy_fea = []
        for i_ in anomaly_idx:
            picked_list = random.sample(all_idx, cluster)
            max_dist = 0
            for j_ in picked_list:
                cur_dist = euclidean(features[i_], features[j_])
                if cur_dist > max_dist:
                    max_dist = cur_dist
                    max_idx = j_
            features[i_] = features[max_idx]
            copy_fea.append(max_idx)
        print('{:d} nodes and their features are constructed as anomalies ({:.0f} edges are added) \n'.format(len(anomaly_idx), num_add_edge))
    else:
        adj_mat = sp.csr_matrix(adj_mat)
        seq_length, nodes, fea = features.shape
        features = torch.reshape(features, (nodes, -1))
        # set single node or 2 nodes are anomalies in each EEG graphs
        p = 1
        q = 2
        # anomaly_idx = [11]
        num_node = adj_mat.shape[0]
        all_idx = list(range(num_node))
        random.shuffle(all_idx)
        anomaly_idx = all_idx[:p * q]
        # print(anomaly_idx)
        anomaly_idx = anomaly_idx[:p * q]

        label = np.zeros((num_node, 1), dtype=np.uint8)
        label[anomaly_idx, 0] = 1
        # print(anomaly_idx)
        for q_ in range(q):
            current_nodes = all_idx
            for i in current_nodes:
                for j in anomaly_idx:
                    adj_mat[j, i] = 1
            adj_mat[current_nodes, current_nodes] = 0
        copy_fea = []
        for i in range(len(anomaly_idx)):
            i_ = anomaly_idx[i]
            picked_list = random.sample(all_idx, num_node)
            # print(picked_list)
            max_dist = 0
            for j_ in picked_list:
                current_dist = euclidean(features[i_], features[j_])
                if current_dist > max_dist:
                    max_dist = current_dist
                    max_idx = j_
            features[i_] = features[max_idx]
            copy_fea.append(max_idx)
    return adj_mat, features, label, copy_fea, anomaly_idx

#4 Generate subgraphs
def sampling_subgraph(dgl_graph, subgraph_size, bf_tr = 3, af_tr = 5):
    """
    Inputs:
        dgl_graph:  NxN
        subgraph_size: num of nodes in a subgraph
    Outputs:
        sub_o
    """
    all_idx = list(range(dgl_graph.number_of_nodes())) #all indexes of total of nodes in a mini-batch
    reduced = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
                                                           max_nodes_per_seed = subgraph_size * bf_tr)
    if_size = 2
    r_time = 10
    sub_o = []

    for i, trace in enumerate(traces):
        sub_o.append(torch.unique(torch.cat(trace),sorted = False).tolist())
        retry_time = 0

        while len(sub_o[i]) < reduced:
            current_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob = 0.9,
                                                                          max_nodes_per_seed = subgraph_size * af_tr)
            sub_o[i] = torch.unique(torch.cat(current_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(sub_o[i]) <= if_size) and (retry_time > r_time):
                sub_o[i] = (sub_o[i] * reduced)
        sub_o[i] = sub_o[i][:reduced]
        sub_o[i].append(i)
    return sub_o



###########################################
# V - HELPER FUNCTIONS
def dense_to_one_hot(labels_dense, num_classes):
    """
    Convert class labels from scalars to one-hot vectors
    """

    num_labels = labels_dense.shape[0]
    idx = np.arange(num_labels) * num_classes
    label_to_one_hot = np.zeros((num_labels, num_classes))
    label_to_one_hot.flat[idx + labels_dense.ravel()] = 1
    return label_to_one_hot


#################################
def threshold_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve
    """
    pre, recall, threshold = precision_recall_curve(y_true, y_prob)
    threshold_inte = []
    f_score = []
    num_threshold = len(threshold)

    for i in range(num_threshold):
        current_f1 = (2 * pre[i] * recall[i]) / \
                     (pre[i] + recall[i])
        if not (np.isnan(current_f1)):
            f_score.append(current_f1)
            threshold_inte.append(threshold[i])

    idx_max = np.argmax(np.array(f_score))
    best_threshold = threshold_inte[idx_max]
    return best_threshold

def dict_metrics(y_pred, y, y_prob=None, file_names=None, average='macro'):
    """
        Compute all metrics for all tasks in EEG-CGS
    """

    metrics_dict = {}
    predicted_dict = defaultdict(list)
    true_dict = defaultdict(list)

    # write into output dictionary
    if file_names is not None:
        for i, file_name in enumerate(file_names):
            predicted_dict[file_name] = y_pred[i]
            true_dict[file_name] = y[i]

    if y is not None:
        metrics_dict['acc'] = accuracy_score(y_true=y, y_pred=y_pred)
        metrics_dict['F1'] = f1_score(y_true=y, y_pred=y_pred, average=average)
        metrics_dict['precision'] = precision_score(y_true=y, y_pred=y_pred, average=average)
        metrics_dict['recall'] = recall_score(y_true=y, y_pred=y_pred, average=average)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics_dict['specificity'] = tn / (tn + fp)
        if y_prob is not None:
            if len(set(y)) <= 2:
                metrics_dict['auroc'] = roc_auc_score(y_true=y, y_score=y_prob)
            metrics_dict['auroc'] = roc_auc_score(y_true=y, y_score=y_prob)
        #scores_dict['auroc'] = roc_auc_score(y_true=y, y_score=y_prob)
    return metrics_dict
