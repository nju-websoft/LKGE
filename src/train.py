from .utils import *
from .model.model_process import *


class Trainer():
    def __init__(self, args, kg, model, optimizer):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        self.train_processor = TrainBatchProcessor(args, kg)
        if self.args.lifelong_name in ['MEAN', 'LAN']:
            self.valid_processor = DevBatchProcessor_MEANandLAN(args, kg)
        else:
            self.valid_processor = DevBatchProcessor(args, kg)

    def run_epoch(self):
        self.args.valid = True
        loss = self.train_processor.process_epoch(self.model, self.optimizer)
        res = self.valid_processor.process_epoch(self.model)
        self.args.valid = False
        return loss, res


