import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

import math
from typing import Optional, List, Union

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_add

def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between true target values and predicted values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Mean squared error.
    """
    squared_errors = (y_true - y_pred) ** 2
    mse = np.mean(squared_errors)
    return mse

def search_data(sequence_length, num_of_depend, label_start_idx,num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_hours, label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target
    num_for_predict: int,the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    hour_sample: np.ndarray   shape is (num_of_hours * points_per_hour, num_of_vertices, num_of_features)
    target: np.ndarray shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None
    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours, label_start_idx, num_for_predict, 1, points_per_hour)
        if not hour_indices:
            return None, None, None, None
        hour_sample = np.concatenate([data_sequence[i: j] for i, j in hour_indices], axis=0)
    
    if num_of_hours > 10:
        return 1;
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(graph_signal_matrix_filename, num_of_hours, num_for_predict, points_per_hour=12):
    '''
    Parameters
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data
    Returns
    feature: np.ndarray, shape is (num_of_samples, num_of_depend * points_per_hour, num_of_vertices, num_of_features)
    target: np.ndarray, shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    #--------------------------------- Read original data 
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (sequence_length, num_of_vertices, num_of_features) (16992, 307, 3)
    
    #---------------------------------
    all_samples = []
    for idx in range(data_seq.shape[0]): # 16992
        sample = get_sample_indices(data_seq, num_of_hours, idx, num_for_predict, points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample #  week_sample, day_sample are None because we are predicting per hour
        #print(target.shape) # hour_sample and target (12, 307, 3)
        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]
        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)
        all_samples.append(sample)#sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)  for i in zip(*all_samples[:split_line1])] #[(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    return training_set, validation_set, testing_set

def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,3), keepdims=True)
    std = train.std(axis=(0,1,3), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    if 'npy' in distance_df_filename:  # false
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        
        #--------------------------------------------- read from here
        import csv
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distaneA[i, j] = distance
        return A, distaneA

def main():
    data = np.load('./data/PeMS04/PEMS04.npz')
    print(data['data'].shape)
    points_per_hour = 12
    num_for_predict = 12
    num_of_hours = 1

    training_set, validation_set, testing_set = read_and_generate_dataset('./data/PeMS04/PEMS04.npz', num_of_hours, 
                                                                        num_for_predict, points_per_hour=points_per_hour)
    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    all_data = {'train': { 'x': train_x_norm, 'target': train_target,'timestamp': train_timestamp},
                'val': {'x': val_x_norm, 'target': val_target, 'timestamp': val_timestamp},
                'test': {'x': test_x_norm, 'target': test_target, 'timestamp': test_timestamp},
                'stats': {'_mean': stats['_mean'], '_std': stats['_std']} }
    
    batch_size = 32
    mean = all_data['stats']['_mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = all_data['stats']['_std'][:, :, 0:1, :]  # (1, 1, 3, 1)
    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    adj_filename = './data/PeMS04/PEMS04.csv'
    num_of_vertices = 307
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, None) #  adj_mx and distance_mx (307, 307)

    rows, cols = np.where(adj_mx == 1)
    edges = zip(rows.tolist(), cols.tolist())

    # gr = nx.Graph()
    # gr.add_edges_from(edges)
    # nx.draw(gr, node_size=3)
    # plt.show()
    # rows, cols = np.where(adj_mx == 1)
    # edges = zip(rows.tolist(), cols.tolist())
    # edge_index_data = torch.LongTensor(np.array([rows, cols])).to(DEVICE)

    mae_predictions = 0
    rmse_predictions = 0
    mape_predictions = 0
    torch.nan_to_num(test_target_tensor, nan=1)
    for i in range(1,test_target_tensor.shape[0]):
        # Select one time frame in the past for prediction
        predicted_data = test_target_tensor[i-1]
        current_data = test_target_tensor[i]
        # Make predictions
        mae_prediction = torch.sum(abs(current_data-predicted_data))
        mae_predictions += mae_prediction

        rmse_prediction = torch.sum((current_data-predicted_data)**2)
        rmse_predictions += rmse_prediction
        
        mape_prediction = torch.sum(abs((current_data-predicted_data)/current_data))
        mape_predictions += mape_prediction
        print(mape_prediction)

    # Calculate the mean squared error (MSE) of the predictions
    print(mape_predictions)
    print(mae_predictions/test_target_tensor.shape[0])
    print(math.sqrt(rmse_predictions/test_target_tensor.shape[0]))
    print(mape_predictions/test_target_tensor.shape[0])

main()