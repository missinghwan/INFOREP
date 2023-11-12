import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nhid_3, nhid_out, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid_1)
        self.gc2 = GraphConvolution(nhid_1, nhid_2)
        self.gc3 = GraphConvolution(nhid_2, nhid_3)
        self.gc4 = GraphConvolution(nhid_3, nhid_out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))

        return x


class Meta_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Meta_Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x



class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nhid_3, nhid_out, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid_out*2, nhid_3)
        self.gc2 = GraphConvolution(nhid_3, nhid_2)
        self.gc3 = GraphConvolution(nhid_2, nhid_1)
        self.gc4 = GraphConvolution(nhid_1, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))

        return x


class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid*2, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x



class MSAD(nn.Module):
    def __init__(self, feat_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_out, meta_size, dropout):
        super(MSAD, self).__init__()

        self.input_encoder = Encoder(feat_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_out, dropout)
        self.meta_encoder = Meta_Encoder(meta_size, hidden_size_out, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_out, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size_out, dropout)

    def forward(self, x, adj, mfeat, madj):
        # encode
        input_encoded = self.input_encoder(x, adj)
        metapath_encoded = self.meta_encoder(mfeat, madj)
        zero_pad = torch.zeros(input_encoded.size(0)-metapath_encoded.size(0), input_encoded.size(1))
        zero_pad = zero_pad.to(torch.device('cuda'))
        metapath_encoded = torch.cat([metapath_encoded, zero_pad])
        graph_mpath_encoded = torch.cat([input_encoded, metapath_encoded], dim=1)

        # decode
        #attr_decoded = self.attr_decoder(input_encoded, adj)
        attr_decoded = self.attr_decoder(graph_mpath_encoded, adj)
        struc_decoded = self.struct_decoder(graph_mpath_encoded, adj)

        # reconstruction results
        return struc_decoded, attr_decoded
