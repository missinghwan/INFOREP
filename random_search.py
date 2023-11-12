import torch, gc
from sklearn.metrics import roc_auc_score
from model import MSAD
from utils import load_anomaly_detection_dataset, load_anomaly_normal_metapath, load_semi_supervised_metapath, load_semi_supervised_dataset
import argparse
import numpy as np

import ray
from ray import air, tune
from ray.air import session
from ray.train.torch import TorchCheckpoint
from ray.tune.schedulers import AsyncHyperBandScheduler


def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # Structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost


def train_MSAD(config):
    gc.collect()
    torch.cuda.empty_cache()

    #adj, attrs, label, adj_label = load_anomaly_detection_dataset()
    #meta_feat, meta_adj = load_anomaly_normal_metapath()

    adj, attrs, label, adj_label = load_semi_supervised_dataset(args.dataset)
    meta_feat, meta_adj = load_semi_supervised_metapath(args.dataset)

    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)
    meta_feat = torch.FloatTensor(meta_feat)
    meta_adj = torch.FloatTensor(meta_adj)

    model = MSAD(feat_size=attrs.size(1), hidden_size_1=1024, hidden_size_2=512, hidden_size_3=128, hidden_size_out=64, meta_size=meta_feat.size(1), dropout=0.3)

    device = torch.device("cuda")
    adj = adj.to(device)
    adj_label = adj_label.to(device)
    attrs = attrs.to(device)
    meta_feat = meta_feat.to(device)
    meta_adj = meta_adj.to(device)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr"))

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        A_hat, X_hat = model(attrs, adj, meta_feat, meta_adj)
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, config.get("alpha"))
        l = torch.mean(loss)
        l.backward()
        optimizer.step()
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

        if epoch % 10 == 0 or epoch == args.epoch - 1:
            model.eval()
            A_hat, X_hat = model(attrs, adj, meta_feat, meta_adj)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
            score = loss.detach().cpu().numpy()

            # Report metrics (and possibly a checkpoint) to Tune
            session.report({"auc": roc_auc_score(label, score)})
            print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Citeseer', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim_out', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--hidden_dim_3', type=int, default=128, help='dimension of hidden embedding (default: 128)')
    parser.add_argument('--hidden_dim_2', type=int, default=512, help='dimension of hidden embedding (default: 512)')
    parser.add_argument('--hidden_dim_1', type=int, default=1024, help='dimension of hidden embedding (default: 1024)')
    parser.add_argument('--meta_dim', type=int, default=64, help='dimension of metapath embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=500, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')

    args = parser.parse_args()

    config = {
        "lr": tune.qloguniform(1e-3, 5e-3, 1e-3),
        "alpha": tune.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
    }

    # for early stopping
    sched = AsyncHyperBandScheduler()

    resources_per_trial = {"cpu": 2, "gpu": 0} # set this for GPUs
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_MSAD),
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="auc",
            mode="max",
            scheduler=sched,
            num_samples=300,
        ),
        param_space=config
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)










