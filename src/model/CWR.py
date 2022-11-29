from .BaseModel import *


class CWR(BaseModel):
    def __init__(self, args, kg):
        super(CWR, self).__init__(args, kg)
        self.initialize_old_data()

    def switch_snapshot(self):
        '''
        Prepare for training on next snapshot.
        '''
        '''1. re-initialize temporal model'''
        self.reinit_param()

        '''2. expand embedding dims for new entities and relations'''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(self.rel_embeddings.weight.data)
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)

    def copyweights_temporal_2_consolidate(self):
        '''
        Merge the temporal model and consolidate model (old)
        Consolidate = 1/2(temporal + consolidate)
        '''
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            old_param = getattr(self, 'old_data_{}'.format(name))
            if "ent" in name:
                updated_param = torch.cat([(param.data[:old_param.size(0)] * old_param + param.data[:old_param.size(0)]) / (1 + old_param), param.data[old_param.size(0):]], dim=0)
            else:
                updated_param = torch.cat(
                    [(param.data[:old_param.size(0)] * old_param + param.data[:old_param.size(0)]) / (1 + old_param),
                     param.data[old_param.size(0):]], dim=0)

            self.register_buffer('old_data_{}'.format(name), updated_param.clone().detach())

    def embedding(self, stage=None):
        '''consolidate model: evaluate, temporal model: learn new facts'''
        if stage == 'Test':
            if self.args.test_FWT:
                old_ent_num = self.old_data_ent_embeddings_weight.size(0)
                old_rel_num = self.old_data_rel_embeddings_weight.size(0)
                return torch.cat((self.old_data_ent_embeddings_weight, self.ent_embeddings.weight[old_ent_num:]), dim=0), torch.cat((self.old_data_rel_embeddings_weight, self.rel_embeddings.weight[old_rel_num:]), dim=0)
            else:
                return self.old_data_ent_embeddings_weight, self.old_data_rel_embeddings_weight
        else:
            return self.ent_embeddings.weight, self.rel_embeddings.weight

    def snapshot_post_processing(self):
        '''after training on a snapshot, merge the consolidate and temporal model'''
        self.copyweights_temporal_2_consolidate()


class TransE(CWR):
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










