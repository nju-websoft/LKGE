from ..utils import *
from copy import deepcopy as dcopy


class KnowledgeGraph():
    def __init__(self, args):
        self.args = args

        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = dict(), dict(), dict(), dict()
        self.relation2inv = dict()

        self.snapshots = {i: SnapShot(self.args) for i in range(int(self.args.snapshot_num))}
        self.load_data()

    def load_data(self):
        '''
        load data from all snapshot file
        '''
        sr2o_all = dict()
        train_all, valid_all, test_all = [], [], []
        for ss_id in range(int(self.args.snapshot_num)):
            self.new_entities = set()
            '''load facts'''
            train_facts = load_fact(self.args.data_path + str(ss_id) + '/' + 'train.txt')
            test_facts = load_fact(self.args.data_path + str(ss_id) + '/' + 'test.txt')
            valid_facts = load_fact(self.args.data_path + str(ss_id) + '/' + 'test.txt')

            '''extract entities & relations from facts'''
            self.expand_entity_relation(train_facts)
            self.expand_entity_relation(valid_facts)
            self.expand_entity_relation(test_facts)

            '''read train/test/valid data'''
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts, order=True)
            test = self.fact2id(test_facts, order=True)

            '''
            Get edge_index and edge_type for GCN"
                edge_index = [[s_1, s_2, ... s_n],[o_1, o_2, ..., o_n]]
                edge_type = [r_1, r_2, ..., r_n]
            '''
            edge_s, edge_r, edge_o = [], [], []
            edge_s, edge_o, edge_r = self.expand_kg(train, 'train', edge_s, edge_o, edge_r, sr2o_all)
            edge_s, edge_o, edge_r = self.expand_kg(valid, 'valid', edge_s, edge_o, edge_r, sr2o_all)
            edge_s, edge_o, edge_r = self.expand_kg(test, 'test', edge_s, edge_o, edge_r, sr2o_all)

            '''prepare data for 're-training' model'''
            train_all += train
            valid_all += valid
            test_all += test

            '''store this snapshot'''
            self.store_snapshot(ss_id, train, train_all, test, test_all, valid, valid_all, edge_s, edge_o, edge_r, sr2o_all)
            self.new_entities.clear()

    def expand_entity_relation(self, facts):
        '''extract entities and relations from new facts'''
        for (s, r, o) in facts:
            '''extract entities'''
            if s not in self.entity2id.keys():
                self.entity2id[s] = self.num_ent
                self.num_ent += 1
            if o not in self.entity2id.keys():
                self.entity2id[o] = self.num_ent
                self.num_ent += 1

            '''extract relations'''
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_rel
                self.relation2id[r + '_inv'] = self.num_rel + 1
                self.relation2inv[self.num_rel] = self.num_rel + 1
                self.relation2inv[self.num_rel + 1] = self.num_rel
                self.num_rel += 2

    def fact2id(self, facts, order=False):
        '''(s name, r name, o name)-->(s id, r id, o id)'''
        fact_id = []
        if order:
            i = 0
            while len(fact_id) < len(facts):
                for (s, r, o) in facts:
                    if self.relation2id[r] == i:
                        fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
                i = i + 2
        else:
            for (s, r, o) in facts:
                fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
        return fact_id

    def expand_kg(self, facts, split, edge_s, edge_o, edge_r, sr2o_all):
        '''expand edge_index, edge_type (for GCN) and sr2o (to filter golden facts)'''
        def add_key2val(dict, key, val):
            '''add {key: value} to dict'''
            if key not in dict.keys():
                dict[key] = set()
            dict[key].add(val)

        for (h, r, t) in facts:
            self.new_entities.add(h)
            self.new_entities.add(t)
            if split == 'train':
                '''edge_index'''
                edge_s.append(h)
                edge_r.append(r)
                edge_o.append(t)
            '''sr2o'''
            add_key2val(sr2o_all, (h, r), t)
            add_key2val(sr2o_all, (t, self.relation2inv[r]), h)
        return edge_s, edge_o, edge_r

    def store_snapshot(self, ss_id, train_new, train_all, test, test_all, valid, valid_all, edge_s, edge_o, edge_r, sr2o_all):
        '''store snapshot data'''
        self.snapshots[ss_id].num_ent = dcopy(self.num_ent)
        self.snapshots[ss_id].num_rel = dcopy(self.num_rel)

        '''train, valid and test data'''
        self.snapshots[ss_id].train_new = dcopy(train_new)
        self.snapshots[ss_id].train_all = dcopy(train_all)
        self.snapshots[ss_id].test = dcopy(test)
        self.snapshots[ss_id].valid = dcopy(valid)
        self.snapshots[ss_id].valid_all = dcopy(valid_all)
        self.snapshots[ss_id].test_all = dcopy(test_all)

        '''edge_index, edge_type (for GCN of MEAN and LAN)'''
        self.snapshots[ss_id].edge_s = dcopy(edge_s)
        self.snapshots[ss_id].edge_r = dcopy(edge_r)
        self.snapshots[ss_id].edge_o = dcopy(edge_o)

        '''sr2o (to filter golden facts)'''
        self.snapshots[ss_id].sr2o_all = dcopy(sr2o_all)
        self.snapshots[ss_id].edge_index = build_edge_index(edge_s, edge_o).to(self.args.device)
        self.snapshots[ss_id].edge_type = torch.cat(
            [torch.LongTensor(edge_r), torch.LongTensor(edge_r) + 1]).to(self.args.device)
        self.snapshots[ss_id].new_entities = dcopy(list(self.new_entities))

        if self.args.lifelong_name in ['LAN', 'MEAN']:
            self.snapshots[ss_id].ent2neigh, self.snapshots[ss_id].edge_index_sample, self.snapshots[ss_id].edge_type_sample, self.snapshots[ss_id].ent_neigh_num = self.snapshots[ss_id].sample_neighbor()


class SnapShot():
    def __init__(self, args):
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.train_new, self.train_all, self.test, self.valid, self.valid_all, self.test_all = list(), list(), list(), list(), list(), list()
        self.edge_s, self.edge_r, self.edge_o = [], [], []
        self.sr2o_all = dict()
        self.edge_index, self.edge_type = None, None
        self.new_entities = []

    def sample_neighbor(self):
        '''sample neighbor for MEAN or LAN'''
        num = 64
        res = []
        triples = self.train_new
        ent2triples = {i:list() for i in range(self.num_ent)}
        edge_index_sample, edge_type_sample = [], []
        ent_neigh_num = torch.zeros(self.num_ent).to(self.args.device)
        for triple in triples:
            h, r, t = triple
            ent2triples[h].append((h, r, t))
            ent2triples[t].append((t, r+1, h))
        for ent in range(self.num_ent):
            ent2triples[ent].append((ent, self.num_rel, ent))
            if len(ent2triples[ent]) > num:
                ent_neigh_num[ent] = num
                samples = [ent2triples[ent][i] for i in np.random.choice(range(len(ent2triples[ent])), num, replace=False)]
            else:
                samples = ent2triples[ent]
                ent_neigh_num[ent] = len(ent2triples[ent])
                for i in range(num - len(ent2triples[ent])):
                    samples.append((self.num_ent, self.num_rel+1, self.num_ent))
            res.append(samples)
            for hrt in samples:
                h, r, t = hrt
                if r == self.num_rel+1:
                    pass
                else:
                    edge_index_sample.append([ent, t])
                    edge_type_sample.append(r)

        return torch.LongTensor(res).to(self.args.device), torch.LongTensor(edge_index_sample).to(self.args.device).t(), torch.LongTensor(edge_type_sample).to(self.args.device), ent_neigh_num
