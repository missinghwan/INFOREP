import torch, gc
from sklearn.metrics import roc_auc_score
from model import *
from utils import *
import argparse


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


def train_MSAD(args):
    gc.collect()
    torch.cuda.empty_cache()

    adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)
    meta_feat, meta_adj = load_anomaly_normal_metapath(args.dataset)

    #adj, attrs, label, adj_label = load_semi_supervised_dataset(args.dataset)
    #meta_feat, meta_adj = load_semi_supervised_metapath(args.dataset)

    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)
    meta_feat = torch.FloatTensor(meta_feat)
    meta_adj = torch.FloatTensor(meta_adj)

    model = MSAD(feat_size=attrs.size(1), hidden_size_1=args.hidden_dim_1, hidden_size_2=args.hidden_dim_2, hidden_size_3=args.hidden_dim_3, hidden_size_out=args.hidden_dim_out, meta_size=meta_feat.size(1), dropout=args.dropout)

    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        attrs = attrs.to(device)
        meta_feat = meta_feat.to(device)
        meta_adj = meta_adj.to(device)
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        A_hat, X_hat = model(attrs, adj, meta_feat, meta_adj)
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

        if epoch % 5 == 0 or epoch == args.epoch - 1:
            model.eval()
            A_hat, X_hat = model(attrs, adj, meta_feat, meta_adj)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
            score = loss.detach().cpu().numpy()
            print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))

    # for visualization
    #model.eval()
    #input_encoded = model.input_encoder(attrs, adj)
    #metapath_encoded = model.meta_encoder(meta_feat, meta_adj)
    #zero_pad = torch.zeros(input_encoded.size(0) - metapath_encoded.size(0), input_encoded.size(1))
    #zero_pad = zero_pad.to(torch.device('cuda'))
    #metapath_encoded = torch.cat([metapath_encoded, zero_pad])
    #z = torch.cat([input_encoded, metapath_encoded], dim=1)
    #visualization(z, label)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Cora', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim_out', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--hidden_dim_3', type=int, default=128, help='dimension of hidden embedding (default: 128)')
    parser.add_argument('--hidden_dim_2', type=int, default=512, help='dimension of hidden embedding (default: 512)')
    parser.add_argument('--hidden_dim_1', type=int, default=1024, help='dimension of hidden embedding (default: 1024)')
    parser.add_argument('--meta_dim', type=int, default=64, help='dimension of metapath embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=500, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.75, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')

    args = parser.parse_args()

    train_MSAD(args)










