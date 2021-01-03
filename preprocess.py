from collections import defaultdict
import os
import sys
import pickle as pk
import scipy.sparse as sp
import numpy as np
import torch
from dataset import Dataset
from lex import LexParser

MAX_SEQ_LEN = 10

user_method_id = {}
item_method_id = {}
class_id = {}
project_id = {}
user_name = []
item_name = []
class_name = []
project_name = []

invocation_matrix = defaultdict(list)
proj_have_class = defaultdict(list)
class_have_method = defaultdict(list)
proj_have_users = []


def get_project_prefix(lines):
    packages = [line.split('#')[0].split('/')[:-2] for line in lines]
    l = min([len(m) for m in packages])
    while l > 0:  # 包名的最长公共前缀作为project prefix
        flag = True
        for m in packages:
            if m[l-1] != packages[0][l-1]:
                flag = False
                break
        if flag:
            break
        else:
            l -= 1
    if l == 0:
        print('error: project have no common prefix')
    return '/'.join(packages[0][:l])


def is_same_project(p1, p2):
    p1 = p1.split('/')[:-2]
    p2 = p2.split('/')[:-2]
    size = min(len(p1), len(p2))
    equals = sum([1 for i in range(size) if p1[i] == p2[i]])
    if size > 0 and equals == 0:
        return False
    if equals == size or equals >= size//2:  # 超过一半前缀相同就认为是同一个project
        return True
    return False


def filter_line(lines):
    # 暂不考虑同一个project内的方法调用
    clean = [line for line in lines if not is_same_project(line[0], line[1])]
    user_cnt = defaultdict(int)
    for line in clean:
        user_cnt[line[0]] += 1
    rm_entries = set([key for key in user_cnt if user_cnt[key] <= 2])
    clean = [line for line in clean if line[0] not in rm_entries]
    return clean


def read_file(name):
    basedir = './data/%s' % name
    file_names = os.listdir(basedir)
    for fname in file_names:
        path = os.path.join(basedir, fname)
        with open(path, 'r') as f:
            '''同一个文件内的属于一个project
               使用文件名作为project id，因为存在相同项目的不同版本, 其prefix相同但文件名不同
               被调用者所属的project暂时没有考虑, 因为从数据中识别出项目名比较困难
            '''
            pname = fname[:-4]  # remove .txt
            project_id[pname] = len(project_id)
            p_id = project_id[pname]
            project_pre = 'PROJECT/'
            print(p_id, pname)

            lines = [line.strip().split('#') for line in f.readlines()]
            clean_lines = filter_line(lines)
            if clean_lines:
                project_pre += clean_lines[0][0].split('/')[0]
            project_name.append(project_pre)
            users = set()
            for raw_pre, suc in clean_lines:
                pre = pname + '/' + raw_pre  # 加上文件名, 区分不同版本的同名API
                if pre not in user_method_id:
                    user_method_id[pre] = len(user_method_id)
                    users.add(user_method_id[pre])
                    user_name.append('UMETHOD/'+raw_pre.split('/')[-1])
                if suc not in item_method_id:  # callee不存在关心不同版本的问题
                    item_method_id[suc] = len(item_method_id)
                    item_name.append('IMETHOD/'+suc.split('/')[-1])
                um_id = user_method_id[pre]
                im_id = item_method_id[suc]
                invocation_matrix[um_id].append(im_id)

                # add class node for caller/user methods
                c_name = '/'.join(pre.split('/')[:-1])  # remove method name
                if c_name not in class_id:
                    class_id[c_name] = len(class_id)
                    proj_have_class[p_id].append(class_id[c_name])  # add a new class to project
                    class_name.append('UCLASS/'+c_name.split('/')[-1])  # remove the project name prefix
                cid = class_id[c_name]
                class_have_method[cid].append('u%d' % um_id)  # 区分是user method还是item method'''

                # add class node for callee/item methods
                c_name = '/'.join(suc.split('/')[:-1])
                if c_name not in class_id:
                    class_id[c_name] = len(class_id)
                    class_name.append('ICLASS/'+c_name.split('/')[-1])
                cid = class_id[c_name]
                class_have_method[cid].append('t%d' % im_id)
            proj_have_users.append(list(users))
            print('project have user methods:', len(users))

    print(len(user_method_id), len(item_method_id), len(class_id))
    assert len(user_name) == len(user_method_id)
    calls = sum([len(val) for val in invocation_matrix.values()])
    print('invocation matrix counts: ', calls)
    rm_entries = set()
    for uid in invocation_matrix:
        if len(invocation_matrix[uid]) <= 1:
            rm_entries.add(uid)
    for uid in rm_entries:
        invocation_matrix.pop(uid, None)

    key = list(invocation_matrix.keys())[0]
    print(user_name[key], invocation_matrix[key])


def build_adj_matrix(nb_proj, nb_class, nb_user, nb_item):
    base_cid = nb_proj
    base_uid = base_cid + nb_class
    base_tid = base_uid + nb_user
    n = base_tid + nb_item
    A = sp.dok_matrix((n, n), dtype=np.float32)


    for pid, val in proj_have_class.items():
        for cid in val:
            A[pid, base_cid + cid] = 1
            A[base_cid + cid, pid] = 1

    for cid, val in class_have_method.items():
        for s in val:
            mid = int(s[1:])
            if s[0] == 'u':
                mid += base_uid
            elif s[0] == 't':
                mid += base_tid
            A[base_cid + cid, mid] = 1
            A[mid, base_cid + cid] = 1
    '''for i, val in enumerate(proj_have_users):
        for uid in val:
            A[i, base_uid + uid] = 1
            A[base_uid + uid, i] = 1'''

    for uid, val in invocation_matrix.items():
        for tid in val:
            A[base_uid + uid, base_tid + tid] = 1
            A[base_tid + tid, base_uid + uid] = 1
    print('build adj matrix')

    def normalize_adj(adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj

    L = normalize_adj(A + sp.eye(A.shape[0]))
    print('normalized adj')
    return L


def strip_user_prefix(name):
    return name[name.index('/')+1:]


def padding_seq(v):
    n = len(v)
    if n >= MAX_SEQ_LEN:
        return v[-MAX_SEQ_LEN:]
    else:
        return v + [0] * (MAX_SEQ_LEN - n)


def build_word_embedding(dataset):
    sents = project_name + class_name + user_name + item_name
    parser = LexParser(sents)
    vec = [parser.vectorize(name) for name in sents]
    max_len = max([len(s) for s in vec])
    print('max sent len:', max_len)
    dataset.lookup_index = torch.LongTensor([padding_seq(s) for s in vec])

    dataset.word_pre_emb = torch.stack([torch.from_numpy(
        emb)for emb in parser.pre_embedding])
    dataset.vocab_sz = len(parser.vocab)

    dataset.user_pre_emb = torch.stack([torch.from_numpy(
        parser.get_embedding(name)) for name in user_name])
    dataset.item_pre_emb = torch.stack([torch.from_numpy(
        parser.get_embedding(name)) for name in item_name])

    class_pre_emb = torch.stack([torch.from_numpy(
        parser.get_embedding(name)) for name in class_name])
    proj_pre_emb = torch.stack([torch.from_numpy(
        parser.get_embedding(name)) for name in project_name])
    dataset.other_pre_emb = torch.cat([proj_pre_emb, class_pre_emb])


def to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset_name):
    name = './tmp/%s-data.pk' % dataset_name
    if not os.path.exists(name):
        print('building dataset from raw file.')
        read_file(dataset_name)
        data = Dataset(len(project_id), len(class_id), len(user_method_id),
                       len(item_method_id), invocation_matrix, proj_have_users)
        data.adj = build_adj_matrix(data.nb_proj, data.nb_class, data.nb_user, data.nb_item)
        build_word_embedding(data)
        with open(name, 'wb') as f:
            pk.dump(data, f)
    with open(name, 'rb') as f:
        data = pk.load(f)
        print('load dataset from disk.')
    # torch sparse tensor cannot be saved to disk
    data.adj = to_torch_sparse_tensor(data.adj)
    return data


def get_calls_distribution(dataset):
    nb_calls = defaultdict(int)
    for pid in range(dataset.nb_proj):
        item_size = [len(dataset.invocation_mx[uid]) for uid in dataset.proj_have_users[pid]]
        for s in item_size:
            nb_calls[s] += 1
    s = 0
    for calls, num in nb_calls.items():
        if calls <= 6:
            print('user having {} API calls count as: {}'.format(calls, num))
        else:
            s += num
    print('user having more than 6 API calls count as: {}'.format(s))


if __name__ == '__main__':
    data = load_data(sys.argv[1])
    L = data.adj.cuda()
    n = L.size()[0]
    x = torch.randn(n, 100).cuda()
    for i in range(10):
        print(i)
        x = torch.sparse.mm(L, x)
