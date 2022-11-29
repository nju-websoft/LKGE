from .BaseModel import *


class Snapshot_only(BaseModel):
    def __init__(self, args, kg):
        super(Snapshot_only, self).__init__(args, kg)
        self.args = args
        self.kg = kg

    def switch_snapshot(self):
        '''prepare for training on next snapshot'''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        self.reinit_param()


class TransE(Snapshot_only):
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



