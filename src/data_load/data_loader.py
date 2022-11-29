from torch.utils.data import Dataset
from src.utils import *


class TestDataset(Dataset):
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg

        '''prepare data for validation and testing'''
        self.valid, self.test = self.build_facts()

    def __len__(self):
        if self.args.valid:
            return len(self.valid[self.args.snapshot])
        else:
            return len(self.test[self.args.snapshot_test])

    def __getitem__(self, idx):
        if self.args.valid:
            ele = self.valid[self.args.snapshot][idx]
        else:
            ele = self.test[self.args.snapshot_test][idx]
        fact, label = torch.LongTensor(ele['fact']), ele['label']
        label = self.get_label(label)

        return fact[0], fact[1], fact[2], label

    @staticmethod
    def collate_fn(data):
        s = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        o = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return s, r, o, label

    def get_label(self, label):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        if self.args.valid:
            y = np.zeros([self.kg.snapshots[self.args.snapshot].num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.kg.snapshots[self.args.snapshot_test].num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid, test = [], []
        for ss_id in range(int(self.args.snapshot_num)):
            valid_, test_ = list(), list()
            if self.args.train_new:
                '''for LKGE and other baselines'''
                for (s, r, o) in self.kg.snapshots[ss_id].valid:
                    valid_.append({'fact': (s, r, o), 'label': self.kg.snapshots[ss_id].sr2o_all[(s, r)]})
            else:
                '''for retraining'''
                for (s, r, o) in self.kg.snapshots[ss_id].valid_all:
                    valid_.append({'fact': (s, r, o), 'label': self.kg.snapshots[ss_id].sr2o_all[(s, r)]})

            if self.args.train_new:
                '''for LKGE and other baselines'''
                for (s, r, o) in self.kg.snapshots[ss_id].valid:
                    valid_.append({'fact': (o, r+1, s), 'label': self.kg.snapshots[ss_id].sr2o_all[(o, r+1)]})
            else:
                '''for retraining'''
                for (s, r, o) in self.kg.snapshots[ss_id].valid_all:
                    valid_.append({'fact': (o, r+1, s), 'label': self.kg.snapshots[ss_id].sr2o_all[(o, r+1)]})

            for (s, r, o) in self.kg.snapshots[ss_id].test:
                test_.append({'fact': (s, r, o), 'label': self.kg.snapshots[ss_id].sr2o_all[(s, r)]})
            for (s, r, o) in self.kg.snapshots[ss_id].test:
                test_.append({'fact': (o, r+1, s), 'label': self.kg.snapshots[ss_id].sr2o_all[(o, r+1)]})
            valid.append(valid_)
            test.append(test_)
        return valid, test


class TrainDatasetMarginLoss(Dataset):
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        self.facts, self.facts_new = self.build_facts()

    def __len__(self):
        if self.args.train_new:  # load the training facts of the i-th snapshots
            return len(self.facts_new[self.args.snapshot])
        else:  # for retraining, load the training facts of first i snapshots
            return len(self.facts[self.args.snapshot])

    def __getitem__(self, idx):
        '''
        :param idx: idx of the training fact
        :return: a positive facts and its negative facts
        '''
        if self.args.train_new:
            ele = self.facts_new[self.args.snapshot][idx]
        else:
            ele = self.facts[self.args.snapshot][idx]
        fact, label = ele['fact'], ele['label']

        '''negative sampling'''
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, None, None

    @staticmethod
    def collate_fn(data):
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        return fact[:,0], fact[:,1], fact[:,2], label

    def build_facts(self):
        '''
        build training data for each snapshots
        :return: training data
        '''
        facts, facts_new = list(), list()
        for ss_id in range(int(self.args.snapshot_num)):
            facts_, facts_new_ = list(), list()
            '''for LKGE and other baselines'''
            for s, r, o in self.kg.snapshots[ss_id].train_new:
                facts_new_.append({'fact':(s, r, o), 'label':1})
                facts_new_.append({'fact': (o, r+1, s), 'label': 1})
            '''for retraining'''
            for s, r, o in self.kg.snapshots[ss_id].train_all:
                facts_.append({'fact':(s, r, o), 'label':1})
                facts_.append({'fact': (o, r+1, s), 'label': 1})
            facts.append(facts_)
            facts_new.append(facts_new_)
        return facts, facts_new

    def corrupt(self, fact):
        '''
        :param fact: positive facts
        :return: positive facts & negative facts ; pos/neg labels.
        '''
        ss_id = self.args.snapshot
        s, r, o = fact
        prob = 0.5

        '''random corrupt subject or object entities'''
        neg_s = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
        neg_o = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
        pos_s = np.ones_like(neg_s) * s
        pos_o = np.ones_like(neg_o) * o
        rand_prob = np.random.rand(self.args.neg_ratio)
        sub = np.where(rand_prob > prob, pos_s, neg_s)
        obj = np.where(rand_prob > prob, neg_o, pos_o)
        facts = [(s, r, o)]

        '''get labels'''
        label = [1]
        for ns, no in zip(sub, obj):
            facts.append((ns, r, no))
            label.append(-1)
        return facts, label


