from .BaseModel import *


class finetune(BaseModel):
    def __init__(self, args, kg):
        super(finetune, self).__init__(args, kg)

    def switch_snapshot(self):
        '''expand embeddings for new entities and relations '''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()

        '''inherit learned parameters'''
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(
            self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(
            self.rel_embeddings.weight.data)
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)


class TransE(finetune):
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
        new_loss = self.new_loss(head, rel, tail, label)
        return new_loss










