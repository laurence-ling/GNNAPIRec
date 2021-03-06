import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import sys
import time
import argparse
import logging
from collections import defaultdict

from preprocess import load_data, get_calls_distribution
from model import GCNRec

logging.basicConfig(format='%(asctime)s-%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
best_suc = [0]*21
best_pre = [0]*21
best_recall = [0]*21
test_config = 'C2.2'


def gvar(indices):
    return torch.LongTensor(indices).to(device)


def train(args):
    dataset = load_data(args.dirname)
    dataset.split_data(test_config)
    adj = dataset.adj.to(device)
    pre_emb = dataset.word_pre_emb.to(device)
    lookup = dataset.lookup_index.to(device)

    logger.info('start training on dataset user:{}, item:{}, other:{}'.format(
                dataset.nb_user, dataset.nb_item, dataset.nb_proj+dataset.nb_class))
    get_calls_distribution(dataset)

    model = GCNRec(dataset.nb_user, dataset.nb_item, dataset.nb_proj+dataset.nb_class,
                   adj, dataset.vocab_sz, lookup, pre_emb).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    num_params = sum([p.numel() for p in model.parameters()])
    logger.info('total model parameters: {}'.format(num_params))

    batch_sz = args.batch_sz
    neg_sz = args.neg_sz
    save_round = args.save_round
    nb_epoches = args.epoch_num
    for i in range(nb_epoches):
        dataset.shuffle_train()
        model.train()
        epoch_loss = 0
        for user, pos_item, neg_item in tqdm(dataset.gen_batch(batch_sz, neg_sz),
                                             total=len(dataset.train)//batch_sz):
            label = np.concatenate((np.ones(batch_sz), np.zeros(batch_sz*neg_sz)))
            loss = model(gvar(user), gvar(pos_item), gvar(neg_item), gvar(label))
            epoch_loss += loss.item()
            if np.isnan(epoch_loss):
                logger.error(epoch_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: {} loss:{}'.format(i, epoch_loss))
        if (i+1) % save_round == 0:
            save_model(model, args.dirname, i+1)
            print('saved model dict')
            eval2(model, dataset)


def eval2(model, dataset):
    test_set = dataset.test_dict
    logger.info('test start. test set size: %d' % len(test_set))
    t1 = time.time()
    model.eval()
    users = np.asarray(list(test_set.keys()))
    top_items = model.get_top_items(gvar(users), k=24).cpu().numpy()
    # 从推荐中去掉已经出现的前4个item, 推荐新的API而不推荐旧的
    used_items = [set(dataset.invocation_mx[uid][:4]) for uid in users]
    items = []
    for i, item in enumerate(top_items):
        rec_item = [tid for tid in item if tid not in used_items[i]]
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
                logger.debug('failed uid %d' % uid)
                logger.debug('GT:{}, REC:{}'.format(test_set[uid], items[i]))
                proj_suc[pid].append(0)
            p = len(intersect) / k
            r = len(intersect) / len(set(test_set[uid]))
            precisions.append(p)
            recalls.append(r)
            proj_pre[pid].append(p)
            proj_recall[pid].append(r)
        suc_rate = len(suc_methods) / len(users)
        #print('----------------------result@%d--------------------------' % k)
        #print('success rate at method level', suc_rate)
        #print('mean precision:{}, mean recall:{}'.format(np.mean(precisions), np.mean(recalls)))
    
        suc_project = [np.mean(val) for val in proj_suc.values()]
        #pres = [np.mean(val) for val in proj_pre.values()]
        #recs = [np.mean(val) for val in proj_recall.values()]
        #print('**********************************************************')
        #print('success rate at project level', np.mean(np.mean(suc_project)))
        #print('mean precision:{}, mean recall:{}'.format(np.mean(pres), np.mean(recs)))
        return suc_rate, np.mean(precisions), np.mean(recalls)

    t2 = time.time()
    logger.info('test end time: {}s'.format(t2 - t1))
    for i in range(1,21):
        suc, pre, rec = res_at_k(i)
        if suc > best_suc[i]:
            best_suc[i] = suc
        if pre > best_pre[i]:
            best_pre[i] = pre
        if rec > best_recall[i]:
            best_recall[i] = rec
        print(i, pre, rec, suc)
        logger.warning('best suc %f, best pre %f,  best recall %f' % (best_suc[i], best_pre[i], best_recall[i]))


def eval(args):
    dataset = load_data(args.dirname)
    dataset.split_data(test_config)
    adj = dataset.adj.to(device)
    model = GCNRec(dataset.nb_user, dataset.nb_item, dataset.nb_proj+dataset.nb_class,
                   adj).to(device)
    load_model(model, args.dirname.split('/')[1], args.epoch_num)
    eval2(model, dataset)


def save_model(model, dir, epoch):
    model_dir = os.path.join('./weight', dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, 'epoch%d.h5' % epoch))


def load_model(model, dir, epoch):
    model_path = './weight/%s/epoch%d.h5' % (dir, epoch)
    assert os.path.exists(model_path), 'Weights not found.'
    model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', default=False, help='eval mode on.')
    parser.add_argument('--dirname', type=str, default='SH_S', help='data set dir.')
    parser.add_argument('--batch_sz', type=int, default=64, help='batch size.')
    parser.add_argument('--neg_sz', type=int, default=2, help='negative sample size.')
    parser.add_argument('--save_round', type=int, default=1, help='save weight per epoch round.')
    parser.add_argument('--epoch_num', type=int, default=30, help='load model weight at epoch number')
    args = parser.parse_args()
    if args.eval:
        eval(args)
    else:
        train(args)
