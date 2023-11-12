import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_anomaly_detection_dataset(dataset, datadir='data'):
# for random search
#def load_anomaly_detection_dataset():
    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    # for random search
    #data_mat = sio.loadmat("C:/Users/user/PycharmProjects/Metapath-based GAE/data/Cora.mat")

    adj = data_mat['Network']
    feat = data_mat['Attributes']
    truth = data_mat['Label']

    #adj = data_mat['A']
    #feat = data_mat['X']
    #truth = data_mat['gnd']

    truth = truth.flatten()

    adj_norm = normalize_adj(adj+sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj+sp.eye(adj.shape[0])
    adj = adj.toarray() # amazon은 주석
    feat = feat.toarray() # amazon은 주석

    return adj_norm, feat, truth, adj


def load_semi_supervised_dataset(dataset):
    # loading node information
    feature_names = ["w_{}".format(ii) for ii in range(3703)]
    column_names = feature_names + ["subject"]
    #node_data = pd.read_csv(f'D:/Anomaly Detection/semi-supervised/{dataset}/{dataset}/{dataset}/{dataset}.content', sep='\t', names=column_names)
    #edge_list = pd.read_csv(f'D:/Anomaly Detection/semi-supervised/{dataset}/{dataset}/{dataset}/{dataset}.cites', sep='\t', names=['target', 'source'])
    node_data = pd.read_csv(f'D:/Anomaly Detection/semi-supervised/{dataset}/{dataset}/{dataset}.content', sep='\t', names=column_names) # citeseer
    edge_list = pd.read_csv(f'D:/Anomaly Detection/semi-supervised/{dataset}/{dataset}/{dataset}_1.cites', sep='\t', names=['target', 'source']) # citeseer

    num_node, num_feature = node_data.shape[0], node_data.shape[1] - 1
    node_index = np.array(node_data.index)

    # mapping the origin node's Id to a new order node's Id that is easier to create the adjacent matrix
    #index_map = {j: i for i, j in enumerate(node_index)}
    #index_map = str(index_map) # citeseer
    #adj = np.zeros((num_node, num_node))

    # create an undirected adjacent matrix
    #for i in range(edge_list.shape[0]):
    #    u = edge_list['target'][i]
    #    v = edge_list['source'][i]
    #    adj[index_map[u], index_map[v]] = 1
    #    adj[index_map[v], index_map[u]] = 1

    # plus adjacent matrix with with identity matrix
    #I = np.eye(num_node)
    #adj_tld = adj + I

    # symmetric normalization
    #rowsum = np.sum(adj_tld, axis=1)
    #r_inv = rowsum ** -0.5
    #r_inv[np.isinf(r_inv)] = 0.  # check devided by 0
    #r_mat_inv = np.diag(r_inv)
    #adj_hat = np.dot(np.dot(r_mat_inv, adj_tld), r_mat_inv)  # r_mat_inv * adj_tld * r_mat_inv

    #adj = np.array(adj_hat)

    G = nx.from_edgelist(edge_list.values.tolist())
    adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    feat = np.array(node_data[feature_names])
    #truth = node_data['subject'].apply(lambda x: '1' if x == 'Rule_Learning' else '0').to_numpy() # cora
    truth = node_data['subject'].apply(lambda x: '1' if x == 'AI' else '0').to_numpy() # citeseer
    truth = truth.flatten()

    return adj_norm, feat, truth, adj


def load_semi_supervised_metapath(dataset):
    feature_names = ["w_{}".format(ii) for ii in range(3703)]
    column_names = feature_names + ["subject"]
    df_mpath = pd.read_csv(f'C:/Users/user/Anaconda3/envs/Anomaly Detection/Semi-supervised AD/{dataset}/metapath/{dataset}_AAN_5n.csv', header=None)
    #node_data = pd.read_csv(f'D:/Anomaly Detection/semi-supervised/{dataset}/{dataset}/{dataset}/{dataset}.content', sep='\t', names=column_names)
    node_data = pd.read_csv(f'D:/Anomaly Detection/semi-supervised/{dataset}/{dataset}/{dataset}.content', sep='\t', names=column_names)
    mpath = df_mpath.to_numpy()
    features = node_data[feature_names]

    G = nx.Graph()

    for path in mpath:
        nx.add_path(G, path)

    meta_adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
    meta_adj = normalize_adj(meta_adj + sp.eye(meta_adj.shape[0]))
    meta_adj = meta_adj.toarray()

    mfeat = []

    for node_idx in np.unique(mpath):
        #mfeat.append(features.loc[node_idx])
        mfeat.append(features.index.values==node_idx)

    mfeat = np.array(mfeat)

    return mfeat, meta_adj


def load_anomaly_normal_metapath(dataset, datadir='data'):
# for ramdom search
#def load_anomaly_normal_metapath():
    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    # for random search
    #data_mat = sio.loadmat("C:/Users/user/PycharmProjects/Metapath-based GAE/data/Cora.mat")

    # Only normals
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Citeseer_metapath/Length_3/5n/Citeseer_NNN_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Cora_metapath/Length_3/5n/Cora_NNN_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/BlogCatalog_metapath/Length_5/1n/BlogCatalog_NNN_1n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Flickr_metapath/Length_5/1n/Flickr_NNNNN_1n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/ACM_metapath/Length_5/5n/ACM_NNNNN_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Pubmed_metapath/Length_5/5n/Pubmed_NNNNN_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Amazon_metapath/Length_5/1n/Amazon_NNNNN_1n_random.csv')

    # 10%
    df_mpath = pd.read_csv(f'{datadir}/{dataset}/{dataset}_ANNA_10percent_3n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/BlogCatalog_metapath/Length_5/1n/BlogCatalog_NAAAA_10percent_1n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Citeseer_metapath/Length_4/5n/Citeseer_ANNA_10percent_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Cora_metapath/Length_3/1n/Cora_AAA_10percent_1n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/ACM_metapath/Length_3/1n/ACM_NNA_10percent_1n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Flickr_metapath/Length_3/5n/Flickr_ANA_10percent_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Pubmed_metapath/Length_5/5n/Pubmed_NNANA_10percent_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Amazon_metapath/Length_5/1n/Amazon_ANNAN_10percent_1n_random.csv')

    # 1, 3, 5%
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/ACM_metapath/Length_3/5n/ACM_ANA_5percent_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Cora_metapath/Length_3/3n/Cora_ANA_5percent_3n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Citeseer_metapath/Length_3/3n/Citeseer_ANA_5percent_3n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Pubmed_metapath/Length_3/5n/Pubmed_ANN_5percent_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/Flickr_metapath/Length_3/5n/Flickr_ANN_5percent_5n_random.csv')
    #df_mpath = pd.read_csv('C:/Users/user/Anaconda3/envs/Anomaly Detection/public dataset/BlogCatalog_metapath/Length_3/5n/BlogCatalog_ANA_5percent_5n_random.csv')

    mpath = df_mpath.to_numpy()
    features = data_mat['Attributes'].toarray()

    #features = data_mat['A'] # for Amazon

    G = nx.Graph()

    for path in mpath:
        nx.add_path(G, path)

    meta_adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
    meta_adj = meta_adj+sp.eye(meta_adj.shape[0])
    meta_adj = normalize_adj(meta_adj+sp.eye(meta_adj.shape[0]))
    meta_adj = meta_adj.toarray()
    mfeat = []

    for node_idx in np.unique(mpath):
        mfeat.append(features[node_idx])

    mfeat = np.array(mfeat)

    return mfeat, meta_adj



def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def visualization(z, label):
    color_list = ["red", "steelblue"]
    #color_list = ["N", "A"]
    colors = [color_list[y - 1] for y in label]

    xs, ys = zip(*TSNE(perplexity=11, init="random", n_iter=5000, verbose=2).fit_transform(z.cpu().detach().numpy()))
    plt.figure(figsize=(3, 2.5), tight_layout=True, dpi=300)
    plt.scatter(xs, ys, s=5, color=colors)
    #plt.savefig('D:/Anomaly Detection/metapath embedding/cora/NNN.png')
    plt.axis(False)
    plt.show()
