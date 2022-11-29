from .BaseModel import *


class PNN(BaseModel):
    def __init__(self, args, kg):
        super(PNN, self).__init__(args, kg)

    def switch_snapshot(self):
        '''
        Prepare next snapshot
        '''
        self.store_old_parameters()
        ent_embeddings, rel_embeddings = self.expand_embedding_size()

        '''inherit learned parameters'''
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(self.rel_embeddings.weight.data)
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)

    def embedding(self, stage=None):
        '''get embeddings'''
        new_ent_embeddings, new_rel_embeddings = self.ent_embeddings.weight, self.rel_embeddings.weight
        if self.args.snapshot > 0:
            '''using old embeddings'''
            old_ent_embeddings, old_rel_embeddings = self.old_data_ent_embeddings_weight, self.old_data_rel_embeddings_weight
            ent_embeddings = torch.cat([old_ent_embeddings, new_ent_embeddings[old_ent_embeddings.size(0):]])
            rel_embeddings = torch.cat([old_rel_embeddings, new_rel_embeddings[old_rel_embeddings.size(0):]])
        else:
            ent_embeddings, rel_embeddings = new_ent_embeddings, new_rel_embeddings
        return ent_embeddings, rel_embeddings

class TransE(PNN):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: s
        :param rel: r
        :param tail: o
        :param label: label of positive (1) or negative (-1) facts
        :return: training loss
        '''
        new_loss = self.new_loss(head, rel, tail, label)
        return new_loss










