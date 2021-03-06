import numpy as np
from collections import defaultdict


class Dataset:
    def __init__(self, np, nc, nu, ni, inv, users, adj=None):
        self.nb_proj = np
        self.nb_class = nc
        self.nb_user = nu
        self.nb_item = ni
        self.invocation_mx = inv
        self.proj_have_users = users
        self.adj = adj
        self.train_dict = {}
        self.test_dict = {}
        self.train = []
        self.test_user2proj = {}
        self.config = (2, 2)
        self.user_pre_emb = []
        self.item_pre_emb = []
        self.other_pre_emb = []
        self.lookup_index = []
        self.word_pre_emb = []
        self.vocab_sz = 0
        #self.split_data('C2.2')

    def split_data(self, conf):
        """conf: C1.1 C1.2"""
        self.config = (int(conf[1]), int(conf[3]))
        np.random.seed(0)
        test_proj_id = set(np.random.choice(range(self.nb_proj), int(self.nb_proj*0.2), replace=False))
        total_users = sum([len(val) for val in self.proj_have_users])
        print('total user methods:{}, test_proj:{}'.format(total_users, test_proj_id))

        def get_test_user(user_id, k):
            gt_users = []
            for uid in user_id:
                if len(set(self.invocation_mx[uid])) <= k:
                    self.train_dict[uid] = self.invocation_mx[uid]
                else:
                    gt_users.append(uid)
            return gt_users

        def add_to_test(gt_users, test_cnt, k):
            for uid in gt_users[-test_cnt:]:
                # add first k invocation for train, and the last for test
                self.train_dict[uid] = self.invocation_mx[uid][:k]
                self.test_dict[uid] = self.invocation_mx[uid][k:]
                self.test_user2proj[uid] = pid
            for uid in gt_users[:-test_cnt]:
                self.train_dict[uid] = self.invocation_mx[uid]

        for pid in test_proj_id:
            size = len(self.proj_have_users[pid])
            # print('test pid and user size', pid, size)
            if self.config[0] == 1:  # remove half user methods
                user_id = self.proj_have_users[pid][: size//2]
            elif self.config[0] == 2:  # keep all user methods
                user_id = self.proj_have_users[pid]
            if self.config[1] == 2:  # retain 4 invocations
                # use 0.2 percent methods per project as active methods for test
                gt_users = get_test_user(user_id, 5)  # users having more than 5 invocations
                test_cnt = len(gt_users) - int(len(gt_users)*0.8)
                add_to_test(gt_users, test_cnt, 4)
            if self.config[1] == 1:  # reserve the first invocation
                gt_users = get_test_user(user_id, 4)
                test_cnt = len(user_id) - int(len(user_id)*0.8)
                add_to_test(gt_users, test_cnt, 1)

        cnt = sum([len(val) for val in self.test_dict.values()])
        print('test set methods count:{}, invocations:{}'.format(len(self.test_dict), cnt))

        for pid in range(self.nb_proj):
            if pid in test_proj_id:
                continue
            for uid in self.proj_have_users[pid]:
                self.train_dict[uid] = self.invocation_mx[uid]

        self.train = [(uid, tid) for uid, val in self.train_dict.items() for tid in val]
        print('train set methods count:{}, invocation: {}'.format(len(self.train_dict), len(self.train)))

    def sample_negative_item(self, uid, num):
        called = set(self.invocation_mx[uid])
        neg_item = []
        while True:
            tid = np.random.randint(0, self.nb_item)
            if tid not in called:
                neg_item.append(tid)
            if len(neg_item) == num:
                break
        return neg_item

    def shuffle_train(self):
        np.random.shuffle(self.train)

    def gen_batch(self, batch_size, neg_size):
        m = len(self.train) // batch_size
        for i in range(m):
            batch = self.train[i*batch_size: (i+1)*batch_size]
            users, pos_items = zip(*batch)
            # neg_items list: k * batch_sz
            neg_items = np.asarray([self.sample_negative_item(uid, neg_size)
                                   for uid in users]).transpose().flatten()
            yield np.asarray(users), np.asarray(pos_items), neg_items


def get_calls_ditribution(dataset):
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
