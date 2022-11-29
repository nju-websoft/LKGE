from .BaseModel import *


class MEAN(BaseModel):
    def __init__(self, args, kg):
        super(MEAN, self).__init__(args, kg)
        self.old_ent_embeddings = None

    def switch_snapshot(self):
        '''
        Prepare for next snapshot.
        '''
        '''expand embeddings for new entities and relations'''
        new_ent_embeddings, new_rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = new_ent_embeddings.weight.data
        new_rel_embeddings = new_rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = self.ent_embeddings.weight.data
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = self.rel_embeddings.weight.data
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)


class TransE(MEAN):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)
        self.gcn = GCN(args, kg)

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: s
        :param rel: r
        :param tail: o
        :param label: label of positive (1) or negative (-1) facts
        :return: training loss
        '''
        s = self.ent_embedding(head, rel)
        r = self.rel_embeddings(rel)
        o = self.ent_embedding(tail, rel)

        s = self.norm_ent(s)
        r = self.norm_rel(r)
        o = self.norm_ent(o)

        score = torch.norm(s + r - o, 1, -1)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        loss = self.margin_loss_func(p_score, n_score, y)/s.size(0)
        return loss


    def ent_embedding(self, ent, query):
        '''
        Get entity embeddings under specific queries.
        '''
        if self.args.valid:
            ent2neigh = self.kg.snapshots[self.args.snapshot].ent2neigh
        else:
            ent2neigh = self.kg.snapshots[self.args.snapshot_test].ent2neigh
        ent_embeddings = self.gcn(x=self.ent_embeddings.weight, r=self.rel_embeddings.weight, ent=ent, ent2neigh=ent2neigh, query=query)
        return ent_embeddings

    def ent_embedding_all(self, query, stage='Train'):
        '''
        Get all entity embeddings under specific query.
        '''
        if stage != 'Test':
            ss = self.kg.snapshots[self.args.snapshot]
            num_ent = self.kg.snapshots[self.args.snapshot].num_ent
        else:
            ss = self.kg.snapshots[self.args.snapshot_test]
            num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
        edge_index, edge_type = ss.edge_index_sample, ss.edge_type_sample
        if self.args.valid:
            ent2neigh = self.kg.snapshots[self.args.snapshot].ent2neigh
        else:
            ent2neigh = self.kg.snapshots[self.args.snapshot_test].ent2neigh
        ent_embeddings = self.gcn(x=self.ent_embeddings.weight, r=self.rel_embeddings.weight, edge_index=edge_index, edge_type=edge_type, ent2neigh=ent2neigh, query=query, num_ent=num_ent)
        return ent_embeddings

    def predict(self, sub, rel, stage='Valid'):
        '''
        Scores all candidate facts for evaluation
        :param head: subject entity id
        :param rel: relation id
        :param stage: object entity id
        :return: scores of all candidate facts
        '''

        '''get entity and relation embeddings'''
        s = self.ent_embedding(sub, rel)
        r = self.rel_embeddings(rel)
        o_all = self.ent_embedding_all(rel, stage)

        s = self.norm_ent(s)
        r = self.norm_rel(r)
        o_all = self.norm_ent(o_all)

        '''s + r - o'''
        pred_o = s + r
        score = 9.0 - torch.norm(pred_o.unsqueeze(1) - o_all, p=1, dim=2)
        score = torch.sigmoid(score)
        return score


class GCN(nn.Module):
    def __init__(self, args, kg):
        super(GCN, self).__init__()
        self.args = args
        self.kg = kg
        self.rel_num = self.kg.snapshots[int(self.args.snapshot_num) - 1].num_rel

        dim = self.args.emb_dim
        self.w_r = get_param((self.kg.snapshots[int(self.args.snapshot_num) - 1].num_rel+1, dim)).to(self.args.device)
        self.loop_edge = get_param((1, dim)).to(self.args.device)

    def forward(self, x, r, ent2neigh=None, query=None, ent=None, edge_index=None, edge_type=None, num_ent=None):
        if ent != None:  # train
            '''get the 1-hop neighbors of ent'''
            neigh = torch.index_select(ent2neigh, 0, ent).reshape(-1, 3)
            neigh_h, neigh_r, neigh_t = neigh[:, 0], neigh[:, 1], neigh[:, 2]
            '''add a self-loop relation'''
            w_r = torch.cat([self.w_r[:self.kg.snapshots[self.args.snapshot].num_rel], self.loop_edge], dim=0)
            '''add zeros-vector for PAD entity and relation'''
            x = torch.cat([x, torch.zeros(1, x.size(1)).to(self.args.device)], dim=0)
            w_r = torch.cat([w_r, torch.zeros(1, x.size(1)).to(self.args.device)], dim=0)

        else:  # valid / test
            if self.args.valid:
                w_r = torch.cat([self.w_r[:self.kg.snapshots[self.args.snapshot].num_rel], self.loop_edge], dim=0)
            else:
                w_r = torch.cat([self.w_r[:self.kg.snapshots[self.args.snapshot_test].num_rel], self.loop_edge], dim=0)
            neigh_r, neigh_t = edge_type, edge_index[1]

        '''prepare neighbor features'''
        tail = torch.index_select(x, 0, neigh_t)
        rel = torch.index_select(w_r, 0, neigh_r)
        neigh = tail - torch.sum(rel*tail, dim=0) * rel

        '''MEAN of all neighbors'''
        if ent != None:
            ent_neigh_num = torch.index_select(self.kg.snapshots[self.args.snapshot].ent_neigh_num,0,ent)
            res_ent = torch.sum(neigh.reshape(ent.size(0), -1, x.size(-1)), dim=1) / ent_neigh_num.unsqueeze(1)
        else:
            res_ent = scatter_mean(src=neigh, index=edge_index[0], dim_size=num_ent, dim=0)
        return res_ent



