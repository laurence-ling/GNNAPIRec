import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConv


class GCNRec(nn.Module):

    def __init__(self, nb_user, nb_item, nb_other, adj,
                 dropout=0.2, margin=1, emb_dim=64):
        super(GCNRec, self).__init__()
        self.nb_user = nb_user
        self.nb_item = nb_item
        self.nb_other = nb_other
        self.margin = margin
        self.dropout = dropout
        self.adj = adj

        self.other_emb = nn.Embedding(nb_other, emb_dim)
        self.user_emb = nn.Embedding(nb_user, emb_dim)
        self.item_emb = nn.Embedding(nb_item, emb_dim)

        self.conv1 = GraphConv(emb_dim, emb_dim)
        self.conv2 = GraphConv(emb_dim, emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)

    def refine_embedding(self):
        all_emb = torch.cat([self.other_emb.weight, self.user_emb.weight, self.item_emb.weight])
        h_emb = []
        conv_emb = F.dropout(self.conv1(all_emb, self.adj),
                             p=self.dropout, training=self.training)
        h_emb.append(conv_emb)
        conv_emb = F.dropout(self.conv2(conv_emb, self.adj),
                             p=self.dropout, training=self.training)
        h_emb.append(conv_emb)
        out_emb = self.linear(conv_emb)
        h_emb.append(out_emb)
        out_emb = torch.cat(h_emb, dim=1)
        return torch.split(out_emb, [self.nb_other, self.nb_user, self.nb_item])

    def get_top_items(self, user, k):
        _, g_user_emb, g_item_emb = self.refine_embedding()
        user_x = F.embedding(user, g_user_emb)
        ratings = user_x.mm(g_item_emb.transpose(0, 1))
        values, indices = ratings.topk(k)
        return indices

    def forward(self, user, pos_item, neg_item, label):
        """
        :param user: (batch_sz,) int
        :param pos_item: (batch_sz,) int
        :param neg_item: (k*batch_sz,) int
        :param adj: laplacian matrix
        :return: loss
        """
        g_other_emb, g_user_emb, g_item_emb = self.refine_embedding()
        # print(g_user_emb.size(), g_item_emb.size(), g_other_emb.size())
        user_x = F.embedding(user, g_user_emb)
        pos_item_x = F.embedding(pos_item, g_item_emb)
        neg_item_x = F.embedding(neg_item, g_item_emb)
        # inner product between user and pos_item (batch_sz,)
        pos_score = user_x.mul(pos_item_x).sum(1)
        # (k, batch_sz, emb_dim)
        neg_item_x = neg_item_x.view(-1, user_x.size()[0], user_x.size(1))
        # (k, batch_sz, emb_dim)x(batch_sz, emb_dim)->(k, batch_sz, emb_dim)
        # sum(2) -> (k, batch_sz)
        neg_score = neg_item_x.mul(user_x).sum(2)
        # (k, batch_sz) - (batch_sz,) -> (k, batch_sz)
        # expectation of negative samples: mean(0) -> (batch_sz,)
        # total loss: sum() -> (scalar)
        rank_loss = torch.mean(neg_score-pos_score+self.margin, dim=0).clamp(min=1e-6).sum()
        ce_loss = self.cross_entropy_loss(pos_score, neg_score, label)
        loss = ce_loss
        return loss

    def log_loss(self, pos_score, neg_score):
        logits = torch.mean(pos_score - neg_score, dim=0)
        return -torch.sum(torch.log(torch.sigmoid(logits)))

    def cross_entropy_loss(self, pos_score, neg_score, label):
        logits = torch.cat([pos_score, torch.flatten(neg_score)])
        loss = F.binary_cross_entropy_with_logits(logits, label.float())
        return loss
