import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import sys
import argparse
from collections import defaultdict

from preprocess import load_data
from model import GCNRec


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def gvar(indices):
    return torch.LongTensor(indices).to(device)


def train(args):
    dataset = load_data(args.dirname)
    dataset.split_data('C2.2')
    adj = dataset.adj.to(device)
    print('dataset', dataset.nb_user, dataset.nb_item, dataset.nb_proj+dataset.nb_class)

    model = GCNRec(dataset.nb_user, dataset.nb_item, dataset.nb_proj+dataset.nb_class,
                   adj).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    batch_sz = args.batch_sz
    neg_sz = args.neg_sz
    save_round = args.save_round
    nb_epoches = 1000
    for i in range(nb_epoches):
        dataset.shuffle_train()
        model.train()
        epoch_loss = 0
        for user, pos_item, neg_item in tqdm(dataset.gen_batch(batch_sz, neg_sz),
                                             total=len(dataset.train)//batch_sz):
            #print(user)
            loss = model(gvar(user), gvar(pos_item), gvar(neg_item))
            epoch_loss += loss.item()
            if np.isnan(epoch_loss):
                print(epoch_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: {} loss:{}'.format(i, epoch_loss))
        if (i+1) % save_round == 0:
            save_model(model, args.dirname.split('/')[1], i+1)
            print('saved model dict')
            eval2(model, dataset)


def eval2(model, dataset):
    test_set = dataset.test_dict
    print('test set size:', len(test_set))
    model.eval()
    users = np.asarray(list(test_set.keys()))
    top_items = model.get_top_items(gvar(users), k=20).cpu().numpy()
    # 从推荐中去掉已经出现的前4个item, 推荐新的API而不推荐旧的
    used_items = [set(dataset.invocation_mx[uid][:4]) for uid in users]
    items = []
    for i, item in enumerate(top_items):
        rec_item = list(set(item).difference(set(used_items[i])))
        items.append(rec_item[:20])

    def res_at_k(k):
        suc_methods = []
        precisions = []
        recalls = []
        proj_suc = defaultdict(list)
        proj_pre = defaultdict(list)
        proj_recall = defaultdict(list)

        for i, uid in enumerate(users):
            pid = dataset.test_user2proj[uid]
            intersect = set(items[i][:k]) & set(test_set[uid])
            if len(intersect) > 0:
                suc_methods.append(uid)
                proj_suc[pid].append(1)
            else:
                proj_suc[pid].append(0)
            p = len(intersect) / k
            r = len(intersect) / len(test_set[uid])
            precisions.append(p)
            recalls.append(r)
            proj_pre[pid].append(p)
            proj_recall[pid].append(r)
        print('----------------------result@%d--------------------------' % k)
        print('success rate at method level', len(suc_methods) / len(users))
        print('mean precision:{}, mean recall:{}'.format(np.mean(precisions), np.mean(recalls)))
        #print(proj_suc)
        #print(proj_pre)
        suc_project = [np.mean(val) for val in proj_suc.values()]
        precisions = [np.mean(val) for val in proj_pre.values()]
        recalls = [np.mean(val) for val in proj_recall.values()]
        print('**********************************************************')
        print('success rate at project level', np.mean(np.mean(suc_project)))
        print('mean precision:{}, mean recall:{}'.format(np.mean(precisions), np.mean(recalls)))

    for k in [1, 10, 20]:
        res_at_k(k)


def eval(dirname):
    dataset = load_data(dirname)
    dataset.split_data('C2.2')
    adj = dataset.adj.to(device)
    model = GCNRec(dataset.nb_user, dataset.nb_item, dataset.nb_proj + dataset.nb_class,
                   adj).to(device)
    load_model(model, dirname.split('/')[1], 590)
    eval2(model, dataset)


def save_model(model, dir, epoch):
    model_dir = os.path.join('./', dir+'-weights')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, 'epoch%d.h5' % epoch))


def load_model(model, dir, epoch):
    model_path = os.path.join('./', dir+'-weights/epoch%d.h5' % epoch)
    assert os.path.exists(model_path), 'Weights not found.'
    model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', default=False, help='eval mode on.')
    parser.add_argument('--dirname', type=str, default='data/SH_S', help='data set dir.')
    parser.add_argument('--batch_sz', type=int, default=128, help='batch size.')
    parser.add_argument('--neg_sz', type=int, default=64, help='negative sample size.')
    parser.add_argument('--save_round', type=int, default=5, help='save weight per epoch round.')
    args = parser.parse_args()
    if args.eval:
        eval(args.dirname)
    else:
        train(args)
