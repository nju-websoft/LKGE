from .BaseModel import *


class SI(BaseModel):
    '''
    We refer to the implementation of https://github.com/GMvandeVen/brain-inspired-replay and https://github.com/adaruna3/continual-kge/blob/main/models/si_models.py.
    '''
    def __init__(self, args, kg):
        super(SI, self).__init__(args, kg)
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.W = {}
        self.p_old = {}
        self.epsilon = 0.1

        '''SI related'''
        self.initialize_old_data()
        self.initialize_W()

    def switch_snapshot(self):
        '''
        Prepare for the training of next snapshot
        SI:
            1. update and expand regularization weight and parameters for new entity and relations;
            2. inherit the model learned from previous snapshot and prepare embeddings for new entities and relations.
        '''
        '''1. update and expand regularization weight and parameters for new entity and relations;'''
        self.update_omega()
        self.initialize_W()
        self.expand_SI_params()

        '''2. inherit the model learned from previous snapshot and prepare embeddings for new entities and relations.'''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(self.rel_embeddings.weight.data)
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)

    def expand_SI_params(self):
        '''
        Expand parameter size for new entities and relations
        '''

        '''the number of entities and relations on this and next snapshot'''
        num_ent_this, num_rel_this = self.kg.snapshots[self.args.snapshot].num_ent, self.kg.snapshots[
            self.args.snapshot].num_rel
        num_ent_next, num_rel_next = self.kg.snapshots[self.args.snapshot + 1].num_ent, self.kg.snapshots[
            self.args.snapshot + 1].num_rel
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '_')
                '''recalculate the importance weights from trajectories'''
                p_current = getattr(self, 'old_data_{}'.format(n))
                try:
                    omega_new = getattr(self, 'omega_{}'.format(n))
                except AttributeError:
                    omega_new = p.detach().clone().zero_()

                '''expand the weights of entity and relation'''
                if 'ent' in n:
                    p_current = torch.cat([p_current, torch.zeros(num_ent_next - num_ent_this, p_current.size(1)).to(self.args.device)],dim=0)
                    self.W[n] = torch.cat([self.W[n], torch.zeros(num_ent_next - num_ent_this, self.W[n].size(1)).to(self.args.device)],dim=0)
                    self.p_old[n] = torch.cat([self.p_old[n],torch.zeros(num_ent_next - num_ent_this, self.p_old[n].size(1)).to(self.args.device)], dim=0)
                    omega_new = torch.cat([omega_new, torch.zeros(num_ent_next - num_ent_this, omega_new.size(1)).to(self.args.device)],dim=0)
                if 'rel' in n:
                    p_current = torch.cat([p_current, torch.zeros(num_rel_next - num_rel_this, p_current.size(1)).to(self.args.device)],dim=0)
                    self.W[n] = torch.cat([self.W[n], torch.zeros(num_rel_next - num_rel_this, self.W[n].size(1)).to(self.args.device)],dim=0)
                    self.p_old[n] = torch.cat([self.p_old[n],torch.zeros(num_rel_next - num_rel_this, self.p_old[n].size(1)).to(self.args.device)], dim=0)
                    omega_new = torch.cat([p_current, torch.zeros(num_rel_next - num_rel_this, omega_new.size(1)).to(self.args.device)],dim=0)

                '''update old parameters data and omega'''
                self.register_buffer('old_data_{}'.format(n), p_current)
                self.register_buffer('omega_{}'.format(n), omega_new)

    def update_W(self):
        '''
        Update the weights for all parameters
        '''
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '_')
                if p.grad is not None:
                    p_old = self.p_old[n]
                    self.W[n].add_(-p.grad * (p.detach() - p_old))
                self.p_old[n] = p.detach().clone()

    def initialize_W(self):
        '''
        Initialize the weight matrix for training
        :return:
        '''
        self.W = {}
        self.p_old = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '_')
                self.W[n] = p.data.clone().zero_()
                self.p_old[n] = p.data.clone()

    def update_omega(self):
        '''
        Before training on a snapshot, update the value of omega
        '''
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '_')

                '''recalculate the importance weights from trajectories'''
                p_prev = getattr(self, 'old_data_{}'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = self.W[n] / (p_change ** 2 + self.epsilon)
                try:
                    omega = getattr(self, 'omega_{}'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                if omega.shape != omega_add.shape:
                    omega_add = torch.cat((omega_add, torch.zeros(omega.size(0)-omega_add.size(0), omega.size(1)).to(self.args.device)),dim=0)
                omega_new = omega + omega_add

                '''update the stored values'''
                self.register_buffer('old_data_{}'.format(n), p_current)
                self.register_buffer('omega_{}'.format(n), omega_new)

    def si_regularization(self):
        '''
        Get regularization loss for all old paramters to constraint the update of old paramters.
        '''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '_')
                    p_prev = getattr(self, 'old_data_{}'.format(n))
                    omega = getattr(self, 'omega_{}'.format(n))
                    num_ent, num_rel = self.kg.snapshots[self.args.snapshot-1].num_ent, self.kg.snapshots[self.args.snapshot-1].num_rel
                    '''only old parameters'''
                    if "ent" in n:
                        losses.append((omega[:num_ent] * (p[:num_ent] - p_prev[:num_ent]) ** 2).sum())
                    elif "rel" in n:
                        losses.append((omega[:num_rel] * (p[:num_rel] - p_prev[:num_rel]) ** 2).sum())
                    else:
                        self.args.logger.info("Unknown model params", "f")
                        exit()
            return sum(losses)
        except AttributeError:
            '''default si loss when no prior snapshot'''
            return torch.tensor(0., device=self.args.device)

    def epoch_post_processing(self, x=None):
        '''
        update the regularization weight of parameters
        '''
        self.update_W()


class TransE(SI):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: new facts loss + regularization loss
        '''
        new_loss = self.new_loss(head, rel, tail, label)
        if self.args.snapshot > 0:
            old_loss = float(self.args.regular_weight) * self.si_regularization()
        else:
            old_loss = 0.0
        return new_loss + old_loss










