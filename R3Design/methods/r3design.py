import torch
import random
import numpy as np
from tqdm import tqdm
from .base_method import Base_method
from .utils import cuda, loss_nll_flatten
from models import R3Design_Model
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from .loss_ntxent import NTXentLoss

alphabet = 'AUCG'
pre_base_pairs = {0: 1, 1: 0, 2: 3, 3: 2}
pre_great_pairs = ((0, 1), (1, 0), (2, 3), (3, 2))

class R3Design(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.ntxent = NTXentLoss(device, temperature=0.5, use_cosine_similarity=True)
        
    def _build_model(self):
        return R3Design_Model(self.args).to(self.device)

    def _cal_recovery(self, dataset, featurizer):
        recovery = []
        S_preds, S_trues = [], []
        for sample in tqdm(dataset):
            sample = featurizer([sample])
            X, S, mask, lengths, clus, ss_pos, ss_pair, names = sample
            X, S, mask, ss_pos = cuda((X, S, mask, ss_pos), device=self.device)
            logits, gt_S = self.model.sample(X=X, S=S, mask=mask)
            log_probs = F.log_softmax(logits, dim=-1)
            
            if self.args.ss_case == 0:
                S_pred = torch.argmax(log_probs, dim=1)
            if self.args.ss_case == 1:
                # secondary sharpen
                ss_pos = ss_pos[mask == 1].long()
                log_probs = log_probs.clone()
                log_probs[ss_pos] = log_probs[ss_pos] / self.args.ss_temp
                S_pred = torch.argmax(log_probs, dim=1)
                
                pos_log_probs = log_probs.softmax(-1)
                for pair in ss_pair[0]:
                    s_pos_a, s_pos_b = pair
                    if s_pos_a == None or s_pos_b == None or s_pos_b >= S_pred.shape[0]:
                        continue
                    
                    if (S_pred[s_pos_a].item(), S_pred[s_pos_b].item()) in pre_great_pairs:
                        continue
                    
                    if pos_log_probs[s_pos_a][S_pred[s_pos_a]] > pos_log_probs[s_pos_b][S_pred[s_pos_b]]:
                        S_pred[s_pos_b] = pre_base_pairs[S_pred[s_pos_a].item()]
                    elif pos_log_probs[s_pos_a][S_pred[s_pos_a]] < pos_log_probs[s_pos_b][S_pred[s_pos_b]]:
                        S_pred[s_pos_a] = pre_base_pairs[S_pred[s_pos_b].item()]
                            
            cmp = S_pred.eq(gt_S)
            recovery_ = cmp.float().mean().cpu().numpy()
            S_preds += S_pred.cpu().numpy().tolist()
            S_trues += gt_S.cpu().numpy().tolist()
            if np.isnan(recovery_): recovery_ = 0.0
            recovery.append(recovery_)
        recovery = np.median(recovery)

        precision, recall, f1, _ = precision_recall_fscore_support(S_trues, S_preds, average=None)
        macro_f1 = f1.mean()
        print('macro f1', macro_f1)
        return recovery

    def train_one_epoch(self, train_loader):
        # Initialize the model
        self.model.train()
        train_sum, train_weights = 0., 0.
        # Start loading and training
        train_pbar = tqdm(train_loader)
        for batch in train_pbar:
            self.optimizer.zero_grad()
            X, oriS, mask, lengths, clus, ss_pos, ss_pair, names = batch
            # X, aug_Xs, aug_idxs, aug_tms, aug_rms, oriS, mask, lengths, clus, ss_pos, names = batch
            X, oriS, mask, lengths, clus, ss_pos = cuda((X, oriS, mask, lengths, clus, ss_pos), device=self.device)
            
            logits_lst, S, graph_prjs = self.model(X, oriS, mask)
            logits = logits_lst[-1]

            # # cluster-level contrastive learning
            # uni_clus = torch.unique(clus)
            # nidxs = torch.zeros_like(clus)
            # for c_idx in uni_clus:
            #     idxs = torch.where(clus == c_idx)[0]
            #     nidxs[idxs] = idxs[torch.randperm(len(idxs))]
            # iidxs = torch.range(0, graph_prjs.shape[0]-1).long()
            # iis = [ii for ii, _ in enumerate(nidxs) if nidxs[ii] != iidxs[ii]]
            # loss_clu_con = self.ntxent(graph_prjs[iis], graph_prjs[nidxs[iis]])

            # # sample-level contrastive learning
            # # select one from augmented data
            # aug_sid = [random.choice(range(len(aug_x))) if len(aug_x) > 0 else None for aug_x in aug_Xs]
            # aug_Xs = [cuda(aug_x[aug_sid[aug_i]], device=self.device) for aug_i, aug_x in enumerate(aug_Xs) if aug_sid[aug_i] != None]
            # aug_tms = [aug_tm[aug_sid[aug_i]] for aug_i, aug_tm in enumerate(aug_tms) if aug_sid[aug_i] != None]
            # aug_rms = [aug_rm[aug_sid[aug_i]] for aug_i, aug_rm in enumerate(aug_rms) if aug_sid[aug_i] != None]
            # aug_Xs = torch.stack(aug_Xs, dim=0)
            # aug_logits_lst, aug_S, aug_graph_prjs = self.model(aug_Xs, oriS[aug_idxs], mask[aug_idxs])
            # aug_logits = aug_logits_lst[-1]

            # loss_sam_con = self.ntxent(aug_graph_prjs, graph_prjs[aug_idxs])
            # if self.args.conf_case == 0:
            #     loss_sam_con = loss_sam_con
            # elif self.args.conf_case == 1:
            #     loss_sam_con = loss_sam_con * torch.exp(-torch.tensor(aug_rms + aug_rms, device=self.device))
            # elif self.args.conf_case == 2:
            #     loss_sam_con = loss_sam_con * torch.exp(torch.tensor(aug_tms + aug_tms, device=self.device) - 1)
            # elif self.args.conf_case == 3:
            #     loss_sam_con = loss_sam_con * (torch.exp(torch.tensor(aug_tms + aug_tms, device=self.device) - 1) + torch.exp(-torch.tensor(aug_rms + aug_rms, device=self.device)))
            # loss_sam_con = loss_sam_con.mean()
                
            # # secondary ss
            if self.args.ss_case == 0:
                basic_loss = 0.
                for cur_logit in logits_lst:
                    basic_loss = basic_loss + self.criterion(cur_logit, S).mean()
            # elif self.args.ss_case == 1:
            #     ss_pos = ss_pos[mask == 1].bool()
            #     basic_loss = (self.criterion(logits[~ss_pos], S[~ss_pos]) + self.criterion(logits[ss_pos] / self.args.ss_temp, S[ss_pos])) / 2.0
            
            # if self.args.aug_log_case == 0:
            #     loss = basic_loss + loss_clu_con * self.args.weigth_clu_con + loss_sam_con * self.args.weigth_sam_con
            # elif self.args.aug_log_case == 1:
            #     loss = (basic_loss + self.criterion(aug_logits, aug_S)) / (1 + len(aug_logits) / len(logits)) + loss_clu_con * self.args.weigth_clu_con + loss_sam_con * self.args.weigth_sam_con
            loss = basic_loss
            loss.backward()
        
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            
            # iter scheduler
            self.scheduler.step()

            log_probs = F.log_softmax(logits, dim=-1)
            loss, _ = loss_nll_flatten(S, log_probs)
            train_sum += torch.sum(loss).cpu().data.numpy()
            train_weights += len(loss)
            train_pbar.set_description('train loss: {:.4f}'.format(loss.mean().item()))
            
        # epoch scheduler
        # self.scheduler.step()
        
        train_loss = train_sum / train_weights
        train_perplexity = np.exp(train_loss)
        return train_loss, train_perplexity
    
    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            valid_sum, valid_weights = 0., 0.
            valid_pbar = tqdm(valid_loader)
            for batch in valid_pbar:
                X, S, mask, lengths, clus, ss_pos, ss_pair, names = batch
                X, S, mask, lengths, clus, ss_pos = cuda((X, S, mask, lengths, clus, ss_pos), device=self.device)
                logits_lst, S, _ = self.model(X, S, mask)
                
                log_probs = F.log_softmax(logits_lst[-1], dim=-1)
                loss, _ = loss_nll_flatten(S, log_probs)
                
                valid_sum += torch.sum(loss).cpu().data.numpy()
                valid_weights += len(loss) 
                valid_pbar.set_description('valid loss: {:.4f}'.format(loss.mean().item()))
        
        valid_loss = valid_sum / valid_weights
        valid_perplexity = np.exp(valid_loss)        
        return valid_loss, valid_perplexity

    def test_one_epoch(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            test_sum, test_weights = 0., 0.
            test_pbar = tqdm(test_loader)
            for batch in test_pbar:
                X, S, mask, lengths, clus, ss_pos, ss_pair, names = batch
                X, S, mask, lengths, clus, ss_pos = cuda((X, S, mask, lengths, clus, ss_pos), device=self.device)
                logits_lst, S, _ = self.model(X, S, mask)
                
                log_probs = F.log_softmax(logits_lst[-1], dim=-1)
                loss, _ = loss_nll_flatten(S, log_probs)
                
                test_sum += torch.sum(loss).cpu().data.numpy()
                test_weights += len(loss)
                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))

            test_recovery = self._cal_recovery(test_loader.dataset, test_loader.featurizer)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
        return test_perplexity, test_recovery