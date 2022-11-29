from .BaseModel import *


class LAN(BaseModel):
    '''
    https://ojs.aaai.org/index.php/AAAI/article/view/4698/4576
    '''
    def __init__(self, args, kg):
        super(LAN, self).__init__(args, kg)
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


class TransE(LAN):
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
        Get entity embeddings for queries.
        '''
        if self.args.valid:
            ent2neigh = self.kg.snapshots[self.args.snapshot].ent2neigh
        else:
            ent2neigh = self.kg.snapshots[self.args.snapshot_test].ent2neigh
        ent_embeddings = self.gcn(x=self.ent_embeddings.weight, r=self.rel_embeddings.weight, ent=ent, ent2neigh=ent2neigh, query=query)

        return ent_embeddings

    def ent_embedding_all(self, query, stage='Train'):
        '''
        Get all entity embeddings for a specific query relation.
        :param query: the query relation
        :return: all entity embeddings under the specific query relation.
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
        self.logic_attn_matrix = self.get_logic_attn().to(self.args.device)
        self.rel_num = self.kg.snapshots[int(self.args.snapshot_num) - 1].num_rel

        dim = self.args.emb_dim
        self.w_r = get_param((self.kg.snapshots[int(self.args.snapshot_num) - 1].num_rel+1, dim)).to(self.args.device)

        '''for attention'''
        self.u_a = get_param((dim, 1)).to(self.args.device)
        self.W_a = get_param((2*dim, dim)).to(self.args.device)
        self.Z = get_param((self.kg.snapshots[int(self.args.snapshot_num) - 1].num_rel, dim)).to(self.args.device)
        self.loop_edge = get_param((1, dim)).to(self.args.device)

    def forward(self, x, r, ent2neigh=None, query=None, ent=None, edge_index=None, edge_type=None, num_ent=None):
        if ent != None:
            '''get neighbors'''
            neigh = torch.index_select(ent2neigh, 0, ent).reshape(-1, 3)
            neigh_h, neigh_r, neigh_t = neigh[:, 0], neigh[:, 1], neigh[:, 2]

            '''neigh ←→ query'''
            index_query = query.unsqueeze(1).tile(1, ent2neigh.size(1)).reshape(-1)

            '''add vector for self-loop edge'''
            w_r = torch.cat([self.w_r[:self.kg.snapshots[self.args.snapshot].num_rel], self.loop_edge], dim=0)

            '''add zero-vectors for PAD entity and relation'''
            x = torch.cat([x, torch.zeros(1, x.size(1)).to(self.args.device)], dim=0)
            w_r = torch.cat([w_r, torch.zeros(1, x.size(1)).to(self.args.device)], dim=0)
        else:
            if self.args.valid:
                w_r = torch.cat([self.w_r[:self.kg.snapshots[self.args.snapshot].num_rel], self.loop_edge], dim=0)
            else:
                w_r = torch.cat([self.w_r[:self.kg.snapshots[self.args.snapshot_test].num_rel], self.loop_edge], dim=0)
            neigh_h, neigh_r, neigh_t = edge_index[0], edge_type, edge_index[1]
            index_query = query[0].unsqueeze(0).tile(1, edge_type.size(0)).reshape(-1)

        '''get neighbor embeddings'''
        tail = torch.index_select(x, 0, neigh_t)
        rel = torch.index_select(w_r, 0, neigh_r)
        neigh = tail - torch.sum(rel*tail, dim=0) * rel

        '''neural attention'''
        q = torch.index_select(self.Z, 0, index_query)
        attn_neigh = torch.exp(torch.matmul(torch.tanh(torch.matmul(torch.cat([q, neigh], dim=1),self.W_a)),self.u_a))
        if ent != None:
            attn_all = torch.sum(attn_neigh.reshape(ent.size(0), ent2neigh.size(1)), dim=1)
            attn_neigh = (attn_neigh.reshape(attn_all.size(0), ent2neigh.size(1)) / attn_all.unsqueeze(1)).reshape(
                -1)
        else:
            attn_all_ = scatter_add(src=attn_neigh, index=edge_index[0], dim_size=num_ent, dim=0).reshape(-1)
            attn_all = torch.index_select(attn_all_, 0, neigh_h)
            attn_neigh = (attn_neigh / attn_all.unsqueeze(1)).reshape(-1)

        '''logic attention'''
        if self.args.valid:
            logic_attn_matrix = self.logic_attn_matrix[:self.kg.snapshots[self.args.snapshot].num_rel][:, :self.kg.snapshots[self.args.snapshot].num_rel]
        else:
            logic_attn_matrix = self.logic_attn_matrix[:self.kg.snapshots[self.args.snapshot_test].num_rel][:, :self.kg.snapshots[self.args.snapshot_test].num_rel]
        attn_rel_num = logic_attn_matrix.size(0)
        logic_attn_matrix = torch.cat((logic_attn_matrix, torch.ones([2, attn_rel_num]).to(self.args.device)), dim=0)
        logic_attn_matrix = torch.cat((logic_attn_matrix, torch.ones([attn_rel_num+2,2]).to(self.args.device)), dim=1)
        logic_attn = logic_attn_matrix[index_query, neigh_r]

        '''attn * neigh'''
        attn_neigh += logic_attn
        neigh = neigh * attn_neigh.unsqueeze(1)

        '''get ent embeddings'''
        if ent != None:
            res_ent = torch.sum(neigh.reshape(ent.size(0), -1, x.size(-1)), dim=1)
        else:
            res_ent = scatter_add(src=neigh, index=edge_index[0], dim_size=num_ent, dim=0)
        return res_ent

    def get_logic_attn(self):
        '''get logic-based attention'''
        rel_num = self.kg.snapshots[int(self.args.snapshot_num)-1].num_rel
        facts = self.kg.snapshots[0].train_new
        attn_matrix = torch.zeros((rel_num,rel_num))
        r2e = dict()
        '''get the neighboring entities of each relation'''
        for fact in facts:
            s, r, o = fact
            if r not in r2e.keys():
                r2e[r] = set()
                r2e[r+1] = set()
            r2e[r].add(s)
            r2e[r].add(o)
            r2e[r+1].add(s)
            r2e[r+1].add(o)
        '''get attention'''
        for i in range(rel_num):
            if i not in r2e.keys():
                continue
            max_len = 1
            for j in range(rel_num):
                if j not in r2e.keys():
                    continue
                if len(r2e[i].intersection(r2e[j])) > max_len:
                    max_len = len(r2e[i].intersection(r2e[j]))
            for j in range(rel_num):
                if j not in r2e.keys():
                    continue
                attn_matrix[i, j] = len(r2e[i].intersection(r2e[j]))/max_len
        return attn_matrix



