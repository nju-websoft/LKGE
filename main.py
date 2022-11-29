from src.utils import *
import argparse
from src.train import *
from src.test import *
from src.config import *
from src.parse_args import args
from src.model.LKGE import TransE as LKGE_TransE
from src.model.GEM import TransE as GEM_TransE
from src.model.EMR import TransE as EMR_TransE
from src.model.CWR import TransE as CWR_TransE
from src.model.LAN import TransE as LAN_TransE
from src.model.PNN import TransE as PNN_TransE
from src.model.finetune import TransE as finetune_TransE
from src.model.MEAN import TransE as MEAN_TransE
from src.model.SI import TransE as SI_TransE
from src.model.Snapshot_only import TransE as Snapshot_TransE
from src.model.EWC import TransE as EWC_TransE
from src.model.retraining import TransE as retraining_TransE
from src.data_load.KnowledgeGraph import KnowledgeGraph
import shutil
from datetime import datetime


class experiment():
    def __init__(self, args):
        self.args = args

        '''1. prepare data file path, model saving path and log path'''
        self.prepare()

        '''2. load data'''
        self.kg = KnowledgeGraph(args)

        '''3. create model and optimizer'''
        self.model, self.optimizer = self.create_model()
        if self.args.lifelong_name == 'GEM':
            self.args.optimizer = self.optimizer

        self.args.logger.info(self.args)

    def create_model(self):
        '''
        Initialize KG embedding model and optimizer.
        return: model, optimizer
        '''
        if self.args.lifelong_name == 'LKGE':
            model = LKGE_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'GEM':
            model = GEM_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'EMR':
            model = EMR_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'LAN':
            model = LAN_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'finetune':
            model = finetune_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'Snapshot':
            model = Snapshot_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'SI':
            model = SI_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'CWR':
            model = CWR_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'PNN':
            model = PNN_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'MEAN':
            model = MEAN_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'EWC':
            model = EWC_TransE(self.args, self.kg)
        elif self.args.lifelong_name == 'retraining':
            self.args.train_new = False
            model = retraining_TransE(self.args, self.kg)
        else:
            self.args.logger.info("Unknown lifelong model name", "f")
            exit()
        model.to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.args.learning_rate), weight_decay=self.args.l2)
        return model, optimizer

    def reset_model(self, model=False, optimizer=False):
        '''
        Reset the model or optimizer
        :param model: If True: reset the model
        :param optimizer: If True: reset the optimizer
        '''
        if model:
            self.model, self.optimizer = self.create_model()
            if self.args.lifelong_name == 'GEM':
                self.args.optimizer = self.optimizer
        if optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.args.learning_rate), weight_decay=self.args.l2)
            if self.args.lifelong_name == 'GEM':
                self.args.optimizer = self.optimizer

    def train(self):
        '''
        Training process
        :return: training time
        '''
        start_time = time.time()
        print("Start Training ===============================>")
        self.best_valid = 0.0
        self.stop_epoch = 0
        trainer = Trainer(self.args, self.kg, self.model, self.optimizer)

        '''Training iteration'''
        for epoch in range(int(self.args.epoch_num)):
            self.args.epoch = epoch
            '''training'''
            loss, valid_res = trainer.run_epoch()
            '''early stop'''
            if self.best_valid < valid_res[self.args.valid_metrics]:
                self.best_valid = valid_res[self.args.valid_metrics]
                self.stop_epoch = max(0, self.stop_epoch-5)
                self.save_model(is_best=True)
            else:
                self.stop_epoch += 1
                self.save_model()
                if self.stop_epoch >= self.args.patience:
                    self.args.logger.info('Early Stopping! Snapshot:{} Epoch: {} Best Results: {}'.format(self.args.snapshot, epoch, round(self.best_valid*100, 3)))
                    break
            '''logging'''
            if epoch % 1 == 0:
                self.args.logger.info('Snapshot:{}\tEpoch:{}\tLoss:{}\tMRR:{}\tHits@10:{}\tBest:{}'.format(self.args.snapshot, epoch,round(loss, 3), round(valid_res['mrr'] * 100, 2), round(valid_res['hits10'] * 100, 2), round(self.best_valid * 100,2)))
        end_time = time.time()
        training_time = end_time - start_time
        return training_time

    def test(self):
        tester = Tester(self.args, self.kg, self.model)
        res = tester.test()
        return res

    def prepare(self):
        '''
        set the log path, the model saving path and device
        :return: None
        '''
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)

        '''set data path'''
        self.args.data_path = args.data_path + args.dataset + '/'
        self.args.save_path = args.save_path + args.dataset + '-' + args.embedding_model + '-' + args.lifelong_name + '-' + args.loss_name

        '''add logging implement to model path for ablation_study'''
        if self.args.lifelong_name == 'LKGE':
            if self.args.using_regular_loss == 'False':
                self.args.save_path = self.args.save_path + '-WO_regularloss'
            if self.args.using_reconstruct_loss == 'False':
                self.args.save_path = self.args.save_path + '-WO_reconstructloss'
            if self.args.using_embedding_transfer == 'False':
                self.args.save_path = self.args.save_path + '-WO_transfer'
            if self.args.using_finetune == 'False':
                self.args.save_path = self.args.save_path + '-WO_finetune'
        if self.args.note != '':
            self.args.save_path = self.args.save_path

        if os.path.exists(args.save_path):
            shutil.rmtree(args.save_path, True)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        self.args.log_path = args.log_path + datetime.now().strftime('%Y%m%d/')
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.args.log_path = args.log_path + args.dataset + '-' + args.embedding_model + '-' + args.lifelong_name + '-' + args.loss_name

        '''add logging implement to log path for ablation_study'''
        if self.args.lifelong_name == 'LKGE':
            if self.args.using_regular_loss == 'False':
                self.args.log_path = self.args.log_path + '-WO_regularloss'
            if self.args.using_reconstruct_loss == 'False':
                self.args.log_path = self.args.log_path + '-WO_reconstructloss'
            if self.args.using_embedding_transfer == 'False':
                self.args.log_path = self.args.log_path + '-WO_embedding_transfer'
            if self.args.using_finetune == 'False':
                self.args.log_path = self.args.log_path + '-WO_finetune'

        '''add additional note to log name'''
        if self.args.note != '':
            self.args.log_path = self.args.log_path + self.args.note

        '''set logger'''
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        logging_file_name = args.log_path + '.log'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        '''set device'''
        torch.cuda.set_device(int(args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device

    def next_snapshot_setting(self):
        '''
        Prepare for next snapshot
        '''
        self.model.switch_snapshot()

    def continual_learning(self):
        '''
        The training process on all snapshots.
        :return:
        '''
        '''prepare'''
        report_results = PrettyTable()
        report_results.field_names = ['Snapshot', 'Time', 'Whole_MRR', 'Whole_Hits@1', 'Whole_Hits@3', 'Whole_Hits@10']
        test_results = []
        training_times = []
        BWT, FWT = [], []
        first_learning_res = []

        '''training process'''
        for ss_id in range(int(self.args.snapshot_num)):
            self.args.snapshot = ss_id  # the training snapshot
            self.args.snapshot_test = ss_id

            '''skip previous snapshots, train on the final snapshot'''
            if self.args.skip_previous == 'True' and self.args.snapshot < int(self.args.snapshot_num) - 1 and self.args.lifelong_name in ['Snapshot', 'retraining']:
                self.next_snapshot_setting()
                self.reset_model(optimizer=True)
                continue

            '''preprocess before training on a snapshot'''
            self.model.pre_snapshot()

            if ss_id > 0:
                if self.args.lifelong_name in ['MEAN', 'LAN']:
                    FWT.append(0)
                else:
                    self.args.test_FWT = True
                    res_before = self.test()
                    FWT.append(res_before['mrr'])
            self.args.test_FWT = False

            '''training'''
            if ss_id == 0 or self.args.lifelong_name not in ['MEAN', 'LAN', 'LKGE'] or (self.args.lifelong_name == 'LKGE' and self.args.using_finetune == 'True'):
                training_time = self.train()
            else:
                training_time = 0

            '''prepare result table'''
            test_res = PrettyTable()
            test_res.field_names = ['Snapshot:'+str(ss_id), 'MRR', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10']

            '''save and reload model'''
            best_checkpoint = os.path.join(self.args.save_path, str(ss_id) + 'model_best.tar')
            self.load_checkpoint(best_checkpoint)

            '''post processing'''
            self.model.snapshot_post_processing()

            '''evaluation'''
            reses = []
            for test_ss_id in range(ss_id+1):
                self.args.snapshot_test = test_ss_id  # the testing snapshot
                res = self.test()
                if test_ss_id == ss_id:
                    first_learning_res.append(res['mrr'])
                test_res.add_row([test_ss_id, res['mrr'], res['hits1'], res['hits3'], res['hits5'], res['hits10']])
                reses.append(res)
            if ss_id == self.args.snapshot_num-1:
                for iid in range(self.args.snapshot_num-1):
                    BWT.append(reses[iid]['mrr']-first_learning_res[iid])

            '''record all results'''
            self.args.logger.info('\n{}'.format(test_res))
            test_results.append(test_res)

            '''record report results'''
            whole_mrr, whole_hits1, whole_hits3, whole_hits10 = self.get_report_result(reses)
            report_results.add_row([ss_id, training_time, whole_mrr, whole_hits1, whole_hits3, whole_hits10])
            training_times.append(training_time)

            '''prepare next snapshot'''
            if self.args.snapshot < int(self.args.snapshot_num) - 1:
                if self.args.lifelong_name in ['Snapshot', 'retraining']:
                    self.reset_model(model=True)
                self.next_snapshot_setting()
                self.reset_model(optimizer=True)
        self.args.logger.info('Final Result:\n{}'.format(test_results))
        self.args.logger.info('Report Result:\n{}'.format(report_results))
        self.args.logger.info('Sum_Training_Time:{}'.format(sum(training_times)))
        self.args.logger.info('Every_Training_Time:{}'.format(training_times))
        self.args.logger.info('Forward transfer: {}  Backward transfer: {}'.format(sum(FWT)/len(FWT), sum(BWT)/len(BWT)))

    def get_report_result(self, results):
        '''
        Get report results of the final model: mrr, hits@1, hits@3, hits@10
        :param results: Evaluation results dict: {mrr: hits@k}
        :return: mrr, hits@1, hits@3, hits@10
        '''
        mrrs, hits1s, hits3s, hits10s, num_test = [], [], [], [], []
        for idx, result in enumerate(results):
            mrrs.append(result['mrr'])
            hits1s.append(result['hits1'])
            hits3s.append(result['hits3'])
            hits10s.append(result['hits10'])
            num_test.append(len(self.kg.snapshots[idx].test))
        whole_mrr = sum([mrr * num_test[i] for i, mrr in enumerate(mrrs)]) / sum(num_test)
        whole_hits1 = sum([hits1 * num_test[i] for i, hits1 in enumerate(hits1s)]) / sum(num_test)
        whole_hits3 = sum([hits3 * num_test[i] for i, hits3 in enumerate(hits3s)]) / sum(num_test)
        whole_hits10 = sum([hits10 * num_test[i] for i, hits10 in enumerate(hits10s)]) / sum(num_test)
        return round(whole_mrr, 3), round(whole_hits1, 3), round(whole_hits3, 3), round(whole_hits10, 3)

    def save_model(self, is_best=False):
        '''
        Save trained model.
        :param is_best: If True, save it as the best model.
        After training on each snapshot, we will use the best model to evaluate.
        '''
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.model.state_dict()
        checkpoint_dict['epoch_id'] = self.args.epoch
        out_tar = os.path.join(self.args.save_path, str(self.args.snapshot) + 'checkpoint-{}.tar'.format(self.args.epoch))
        torch.save(checkpoint_dict, out_tar)
        if is_best:
            best_path = os.path.join(self.args.save_path, str(self.args.snapshot) + 'model_best.tar')
            shutil.copyfile(out_tar, best_path)

    def load_checkpoint(self, input_file):
        if os.path.isfile(input_file):
            logging.info('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file, map_location="cuda:{}".format(self.args.gpu))
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info('=> no checkpoint found at \'{}\''.format(input_file))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    same_seeds(int(args.seed))
    config(args)
    E = experiment(args)
    E.continual_learning()




