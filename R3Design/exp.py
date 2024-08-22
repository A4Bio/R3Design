import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import logging
import pickle
import json
import torch
import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from constants import method_maps
from API import Recorder
from utils import *

import wandb
import logging
import pickle
import json
import torch
from parsers import create_parser



from constants import method_maps
from API import Recorder
from utils import *


class Exp:
    def __init__(self, args):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(parent_dir,self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        # self._get_data()
        # build the method
        self._build_method()

    def _build_method(self):
        # steps_per_epoch = len(self.train_loader)
        steps_per_epoch = 1
        self.method = method_maps[self.args.method](self.args, self.device, steps_per_epoch)

    def _get_data(self):
        self.train_loader, self.valid_loader, self.test_loader = get_dataset(self.config)

    def _save(self, name=''):
        torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))
        fw = open(osp.join(self.checkpoints_path, name + '.pkl'), 'wb')
        state = self.method.scheduler.state_dict()
        pickle.dump(state, fw)

    def _load(self, epoch):
        self.method.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')))
        fw = open(osp.join(self.checkpoints_path, str(epoch) + '.pkl'), 'rb')
        state = pickle.load(fw)
        self.method.scheduler.load_state_dict(state)

    def train(self):
        recorder = Recorder(verbose=True)
        for epoch in range(self.args.epoch):
            train_loss, train_perplexity = self.method.train_one_epoch(self.train_loader)
            if epoch % self.args.log_step == 0:
                with torch.no_grad():
                    valid_loss, valid_perplexity = self.valid(epoch)
                    if epoch % (self.args.log_step * 100) == 0 and epoch > 0:
                        self._save(name=str(epoch))
                print_log('Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Train Perp: {3:.4f} Valid Loss: {4:.4f} Valid Perp: {5:.4f}\n'.format(epoch + 1, len(self.train_loader), train_loss, train_perplexity, valid_loss, valid_perplexity))
                recorder(valid_loss, self.method.model, self.path)
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path))

    def valid(self, epoch):
        valid_loss, valid_perplexity = self.method.valid_one_epoch(self.valid_loader)
        print_log('Valid Perp: {0:.4f}'.format(valid_perplexity))
        return valid_loss, valid_perplexity

    def test(self):
        test_perplexity, test_recovery = self.method.test_one_epoch(self.test_loader)
        print_log('Test Perp: {0:.4f}, Test Rec: {1:.4f}\n'.format(test_perplexity, test_recovery))
        return test_perplexity, test_recovery
    
if __name__ == '__main__':
    args = create_parser()
    config = args.__dict__

    
    # default_params = load_config(osp.join('./configs', args.method + '.py' if args.config_file is None else args.config_file))
    # config.update(default_params)

    exp = Exp(args)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<')
    test_perp, test_rec = exp.test()

    if args.wandb:
        wandb.finish()