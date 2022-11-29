from .BaseModel import *
import quadprog

class GEM(BaseModel):
    '''
    We refer to the implementation of https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py.
    '''
    def __init__(self, args, kg):
        super(GEM, self).__init__(args, kg)
        self.mse_loss = nn.MSELoss(size_average=False)

        self.margin = 0.5

        self.args.n_memories = 5000
        self.gpu = self.args.device

        self.memory_data = []

        '''initialize gradient memory'''
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), 2)
        self.grads = self.grads.to(self.args.device)

    def pre_snapshot(self):
        '''prepare for training on this snapshot'''
        if self.args.snapshot == 0:
            self.initialize_memory()
        else:
            self.update_memory()

    def initialize_memory(self):
        train_data = self.kg.snapshots[0].train_new
        self.memory_data = random.sample(train_data, self.args.n_memories)

    def update_memory(self):
        '''randomly replace half of old facts'''
        random.shuffle(self.memory_data)
        train_data = self.kg.snapshots[self.args.snapshot].train_new
        self.memory_data = self.memory_data[:self.args.n_memories // 2] + random.sample(train_data,self.args.n_memories // 2)


    def switch_snapshot(self):
        '''prepare for training on next snapshot'''
        self.store_old_parameters()
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(self.rel_embeddings.weight.data)
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)

    def store_grad(self, grads, grad_dims, tid=2):
        '''store the gradients'''
        grads[:, tid].fill_(0.0)
        cnt = 0
        for param in self.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en, tid].copy_(param.grad.data.view(-1)[:en - beg])
            cnt += 1

    def overwrite_grad(self, newgrad, grad_dims):
        '''overwrite new grad'''
        cnt = 0
        for param in self.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            cnt += 1

    def project2cone2(self, gradient, memories, margin=0.5, eps=1e-3):
        '''change the new grad using GEM project function'''
        memories_np = memories.cpu().t().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1, 1))

    def replay(self, x, label):
        '''
        use stored facts to constrain new updating
        '''
        '''1. compute gradient on previous tasks'''
        self.zero_grad()
        pt_triples, pt_label = self.corrupt(self.memory_data)
        pt_triples, pt_label = torch.LongTensor(pt_triples).to(self.args.device), torch.Tensor(pt_label).to(self.args.device)
        head, rel, tail = pt_triples[:, 0], pt_triples[:, 1], pt_triples[:, 2]
        ptloss = self.new_loss(head, rel, tail, pt_label)
        ptloss.backward()
        self.store_grad(self.grads, self.grad_dims, 0)

        '''2. compute the grad on the current mini-batch'''
        self.zero_grad()
        head, rel, tail = x[:, 0], x[:, 1], x[:, 2]
        loss = self.new_loss(head, rel, tail, label)

        ''' 3. check if gradient violates constraints'''
        self.store_grad(self.grads, self.grad_dims, 1)
        index = torch.cuda.LongTensor([0, 1]).to(self.args.device)  # 0: old facts, 1: new facts
        dotp = torch.mm(self.grads[:, 1].unsqueeze(0), self.grads.index_select(1, index))
        if (dotp < 0).sum() != 0:
            self.project2cone2(self.grads[:, 1].unsqueeze(1),
                               self.grads.index_select(1, index), self.margin)
            # copy gradients back
            self.overwrite_grad(self.grads[:, 1], self.grad_dims)
        return loss

    def corrupt(self, triples):
        '''generate negative triples for old facts'''
        triples = triples
        ss_id = self.args.snapshot
        label = []
        triples_ = []
        for triple in triples:
            s, r, o = triple[0], triple[1], triple[2]
            prob = 0.5
            '''random corrupt subject or object entity'''
            neg_s = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            neg_o = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            pos_s = np.ones_like(neg_s) * s
            pos_o = np.ones_like(neg_o) * o
            rand_prob = np.random.rand(self.args.neg_ratio)
            sub = np.where(rand_prob > prob, pos_s, neg_s)
            obj = np.where(rand_prob > prob, neg_o, pos_o)
            triples_.append((s, r, o))
            label.append(1)
            for ns, no in zip(sub, obj):
                triples_.append((ns, r, no))
                label.append(-1)
            return triples_, label



class TransE(GEM):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: new facts loss
        '''
        x = torch.cat([head.unsqueeze(1), rel.unsqueeze(1), tail.unsqueeze(1)], dim=1)
        if self.args.snapshot > 0:
            new_loss = self.replay(x, label)
        else:
            new_loss = self.new_loss(head, rel, tail, label)
        return new_loss




