from ..utils import *
from ..data_load.data_loader import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor, from_numpy, no_grad, save, load, arange
from torch.autograd import Variable

class TrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        '''prepare data'''
        self.dataset = TrainDatasetMarginLoss(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=int(self.args.batch_size),
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),  # use seed generator
                                      pin_memory=True)

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            '''get loss'''
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                       br.to(self.args.device),
                                       bt.to(self.args.device),
                                       by.to(self.args.device) if by is not None else by).float()

            '''update'''
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            '''post processing'''
            model.epoch_post_processing(bh.size(0))
        return total_loss


class DevBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = 100
        '''prepare data'''
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = dict()
        sr2o = self.kg.snapshots[self.args.snapshot].sr2o_all
        '''start evaluation'''
        for step, batch in enumerate(self.data_loader):
            sub, rel, obj, label = batch
            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            pred = model.predict(sub, rel, stage=stage)

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results


class DevBatchProcessor_MEANandLAN():
    '''
    To save memory, we collect the queries with the same relation and then perform evaluation.
    '''
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = 1
        '''prepare data'''
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = dict()
        '''start evaluation'''
        sub, rel, obj, label = None, None, None, None

        for step, batch in enumerate(self.data_loader):
            sub_, rel_, obj_, label_ = batch
            if sub == None:
                sub, rel, obj, label = sub_, rel_, obj_, label_
                continue
            elif rel[0] == rel_ and rel.size(0) <= 50:
                sub = torch.cat((sub, sub_), dim=0)
                rel = torch.cat((rel, rel_), dim=0)
                obj = torch.cat((obj, obj_), dim=0)
                label = torch.cat((label, label_), dim=0)
                continue

            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            pred = model.predict(sub, rel, stage=stage)

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
            sub, rel, obj, label = None, None, None, None

        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results