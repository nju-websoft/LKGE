import torch
from .utils import *
from .model.model_process import *


class Tester():
    def __init__(self, args, kg, model):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.model = model
        if self.args.lifelong_name in ['MEAN', "LAN"]:
            self.test_processor = DevBatchProcessor_MEANandLAN(args, kg)
        else:
            self.test_processor = DevBatchProcessor(args, kg)

    def test(self):
        self.args.valid = False
        res = self.test_processor.process_epoch(self.model)
        return res