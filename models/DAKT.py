import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from utils.inc_net import IncrementalNet, CosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
import os
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import time
import copy
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from convs.linears import SimpleLinear
import cvxpy as cp
np.set_printoptions(linewidth=200, precision=6, suppress=True)
import random
import math
import csv

import warnings
warnings.filterwarnings("ignore", message="Failed toimport warnings")
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

"""

"""
EPSILON = 1e-8

class DAKT(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if self.args["cosine"]:
            if self.args["dataset"] == "cub200" or self.args["dataset"] == "cars":
                self._network = CosineIncrementalNet(args, True)
            else:
                self._network = CosineIncrementalNet(args, False)
        else:
            if self.args["dataset"] == "cub200" or self.args["dataset"] == "cars":
                self._network = IncrementalNet(args, True)
            else:
                self._network = IncrementalNet(args, False)

        self._protos = []
        self.granularball_list_obj = None

        self.init_epoch = args['init_epoch']
        self.init_lr = args['init_lr']
        self.init_milestones = args['init_milestones']
        self.init_lr_decay = args['init_lr_decay']
        self.init_weight_decay = args['init_weight_decay']
        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.milestones = args['milestones']
        self.lrate_decay = args['lrate_decay']
        self.batch_size = args['batch_size']
        self.weight_decay = args['weight_decay']
        self.num_workers = args['num_workers']

        self.w_kd = args.get('w_kd', 10)
        self.w_fd = args.get('w_fd', 0.1)
        self.w_smp_dif = args.get('w_smp_dif', 0.1)
        self.T = args.get('T', 2)
        
        self.base_value_of_cls_atten = args.get('base_value_of_cls_atten', 0.5)
        self.sim_threshold = args.get('sim_threshold', 0.5)
        self.w_l21 = args.get('w_l21', 0.01)

        self.use_fea_attn = args.get('use_fea_attn', True)
        self.use_cls_attn = args.get('use_cls_attn', True)
        self.use_smp_attn = args.get('use_smp_attn', True)

        self.use_past_model = args['use_past_model']
        self.save_model = args['save_model']
        self.model_dir = args['model_dir']
        self.dataset = args['dataset']
        self.init_cls = args['init_cls']
        self.increment = args['increment']
        self._process_id = args['process_id']

    def after_task(self):
        self._old_network = self._network.copy().freeze() 
        self._known_classes = self._total_classes 
        if self.save_model: #  or self._cur_task == 0
            path = self.model_dir + "{}/{}".format(self.dataset, self.args['seed'])
            if not os.path.exists(path):
                os.makedirs(path)
            self.save_checkpoint("{}/{}_{}".format(path, self.init_cls, self.increment))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._per_task_cls = data_manager.get_task_size(self._cur_task)
        if self.args["cosine"]:
            self._network.update_fc(self._total_classes, self._cur_task)
        else:
            self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        self.shot = None
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)   # , pin_memory = True

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        if self._old_network is not None:
            self._old_network.to(self._device)
        model_dir = "{}/{}/{}/{}_{}_{}.pkl".format(self.args["model_dir"], self.args["dataset"], self.args['seed'],
                                                   self.args["init_cls"], self.args["increment"], self._cur_task)
        # print(model_dir)
        if self._cur_task == 0:
            if self.use_past_model and os.path.exists(model_dir):
                self._network.load_state_dict(torch.load(model_dir)["model_state_dict"], strict=True)
                self._network.to(self._device)
            else:
                self._network.to(self._device)
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.init_milestones, gamma=self.init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            if self.use_past_model and os.path.exists(model_dir):
                self._network.load_state_dict(torch.load(model_dir)["model_state_dict"], strict=True)
                self._network.to(self._device)
            else:
                self._network.to(self._device)
                optimizer = optim.SGD(self._network.parameters(), lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)
                self._update_representation(train_loader, test_loader, optimizer, scheduler)
            self._update_memory(train_loader)
        self._build_protos()

    def _retrain_drift_estimator(self, train_loader):
        if hasattr(self._network, "module"):
            _network = self._network.module
        else:
            _network = self._network
        
        optimizer = optim.Adam(self.drift_estimator.parameters(), lr=0.001)
        for epoch in range(20):
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                feats_old = self._old_network(inputs)["features"]
                feats_new = _network(inputs)["features"]
                x_proj = self.drift_estimator(feats_old)

                loss = torch.nn.MSELoss()(x_proj, feats_new)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _update_memory(self, train_loader):
        if self.args["dataset"] == "cub200" or self.args["dataset"] == "cars":
            self.drift_estimator.kaiming_init_W()
            self._retrain_drift_estimator(train_loader)

        with torch.no_grad():
            for cls_index in range(0, self._known_classes):
                tmp = self.drift_estimator(torch.tensor(self._protos[cls_index]).to(self._device).to(torch.float32))
                self._protos[cls_index] = (self._protos[cls_index] + tmp.cpu().numpy()) / 2
        
    def _build_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),source='train', mode='test', shot=self.shot, ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0)
            self._protos.append(class_mean)

    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.init_epoch), colour='red', position=self._process_id, dynamic_ncols=True, ascii=" =", leave=True)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            L_all = 0.0
            L_new_cls = 0.0
            L_cont = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                features = outputs["features"]
                logits = self._network.fc(features)['logits']

                loss_new_cls = F.cross_entropy(logits, targets)
                L_new_cls += loss_new_cls.item()

                loss = loss_new_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                L_all += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.init_epoch,
                                    L_all / len(train_loader),
                                    L_new_cls / len(train_loader),
                                    L_cont / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.init_epoch,
                                    L_all / len(train_loader),
                                    L_new_cls / len(train_loader),
                                    L_cont / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        if hasattr(self._network, "module"):
            _network = self._network.module
        else:
            _network = self._network
        self._network.eval()
        class DriftEstimator(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.W = nn.Parameter(torch.eye(in_features, out_features))
            
            def kaiming_init_W(self):
                nn.init.kaiming_uniform_(self.W, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu")
            def forward(self, x):
                if x.dim() == 1:
                    x=x.unsqueeze(0)
                
                out = x @ self.W
                if out.size(0) == 1:
                    return out.squeeze(0)
                return out

        P_numpy = np.stack(self._protos)
        P = torch.from_numpy(P_numpy).to(self._device)

        P_shot = np.stack(self._protos[-self._per_task_cls:])
        n, d = P_shot.shape
        M = cp.Variable((n, n))
        reconstruction = cp.norm(P_shot - M @ P_shot, 'fro')**2
        l21 = cp.sum(cp.norm(M, axis=1))
        constraints = [cp.diag(M) == 0]
        problem = cp.Problem(
            cp.Minimize(reconstruction + self.w_l21 * l21),
            constraints
        )
        problem.solve(solver=cp.SCS)
        M_opt = M.value
        # print(M_opt)

        observer = DriftEstimator(self.feature_dim, self.feature_dim).to(self._device)
        prog_bar = tqdm(range(self.epochs), colour='red', position=self._process_id, dynamic_ncols=True, ascii=" =", leave=True)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            L_all = 0.0
            L_new_cls = 0.0
            L_kd = 0.0
            L_fd = 0.0
            L_plasticity = 0.0
            correct, total = 0, 0
            
            X1TX1 = torch.eye(self.feature_dim).to(self._device) * 1e-9
            X1TX2 = 0
            
            with torch.no_grad():
                P_tmp = observer(P)
                dists = torch.norm(P_tmp - P, dim=1)
                
                proto_coef = torch.ones(dists.shape[0], device=self._device)
                consider_cls_num = self._per_task_cls * 3
                if len(self._protos) > consider_cls_num:
                    recent_dists = dists[-consider_cls_num:]
                    d_min, d_max = recent_dists.min(), recent_dists.max()
                    proto_norm = (recent_dists - d_min) / (d_max - d_min + EPSILON)
                    proto_sig = torch.sigmoid(proto_norm)
                    proto_coef[-consider_cls_num:] = self.base_value_of_cls_atten + proto_sig
                    proto_coef[-consider_cls_num:] = proto_coef[-consider_cls_num:] / proto_coef[-consider_cls_num:].mean()
                else:
                    d_min, d_max = dists.min(), dists.max()
                    proto_norm = (dists - d_min) / (d_max - d_min + EPSILON)
                    proto_sig = torch.sigmoid(proto_norm)
                    proto_coef = self.base_value_of_cls_atten + proto_sig
                    proto_coef = proto_coef / proto_coef.mean()
                
                class_specific_difficulty = proto_coef.to(self._device)

                vec1 = np.ones((1, P_shot.shape[0]))
                P_tmp_numpy = P_tmp.cpu().numpy()
                E = np.abs(M_opt @ P_numpy[-self._per_task_cls:] - M_opt @ P_tmp_numpy[-self._per_task_cls:])
                feature_difference = (vec1 @ E) / E.shape[0]

                s = np.log1p(feature_difference)
                s_min, s_max = s.min(), s.max()
                w = 1.0 + (s - s_min) / (s_max - s_min + EPSILON)
                feat_weight_np = w.astype(np.float32)
                feature_wise_attn = torch.from_numpy(feat_weight_np).float().to(self._device)
                
            
            for i, (sample_idx, inputs, targets) in enumerate(train_loader):
                loss_clf, loss_kd, loss_fd, loss_plasticity = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = _network(inputs)
                features = outputs["features"]
                logits = outputs["logits"]
                with torch.no_grad():
                    outputs_old = self._old_network(inputs)
                    fea_old = outputs_old["features"]
                    logits_old = outputs_old["logits"].detach()

                    X1 = fea_old.detach()
                    X2 = features.detach()
                    X1TX1 += X1.T @ X1
                    X1TX2 += X1.T @ X2

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                L_new_cls += loss_clf.item()
                
                if self.use_cls_attn:
                    loss_kd = _difficulty_aware_kd(logits[:, :self._known_classes], logits_old[:, :self._known_classes], self.T, class_specific_difficulty) * self.w_kd
                else:
                    loss_kd = _KD_loss(logits[:, :self._known_classes], logits_old[:, :self._known_classes], self.T) * self.w_kd

                L_kd += loss_kd.item()

                feat_new = F.normalize(features, p=2, dim=1)
                feat_old = F.normalize(fea_old, p=2, dim=1)

                
                diff = feat_new - feat_old
                if self.use_fea_attn == False:
                    w_feat = torch.ones(1, feat_new.size(1), device=feat_new.device, dtype=feat_new.dtype)
                    loss_fd = (diff.pow(2) * w_feat).mean() * self.w_fd
                else:
                    loss_fd = (diff.pow(2) * feature_wise_attn).mean() * self.w_fd
                L_fd += loss_fd.item()
                
                with torch.no_grad():
                    dists_cos_local = 1 - F.cosine_similarity(P_tmp, P, dim=1)
                    sel_mask = dists_cos_local < dists_cos_local.mean()
                    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1) if sel_mask.any() else torch.tensor([], device=self._device, dtype=torch.long)

                if sel_idx.numel() > 0:
                    feats_norm = F.normalize(features, p=2, dim=1)
                    protos_norm = F.normalize(P_tmp, p=2, dim=1)
                    sims = feats_norm @ protos_norm.t()
                    sims_sel = sims[:, sel_idx]
                    sim_threshold = self.sim_threshold
                    mask = (sims_sel > sim_threshold).float()
                    masked = sims_sel * mask
                    sample_difficulty, _ = masked.max(dim=1)
                    sample_difficulty = torch.clamp(sample_difficulty, min=0.0)

                    nz = (sample_difficulty > 0).sum().float()
                    if nz > 0 and self.use_cls_attn:
                        raw = sample_difficulty.sum() / (nz + EPSILON)
                        loss_plasticity = -torch.log(1.0 - raw + EPSILON) * self.w_smp_dif

                L_plasticity += loss_plasticity.item() 

                loss = loss_clf + loss_kd + loss_fd + loss_plasticity

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                L_all += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

            scheduler.step()

            with torch.no_grad():
                W = torch.inverse(X1TX1) @ X1TX2
                observer.W.data.copy_(W.float())

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_kd {:.3f}, L_fd {:.3f}, L_pals {:.3f}, Train_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.epochs,
                                    L_all / len(train_loader),
                                    L_new_cls / len(train_loader),
                                    L_kd / len(train_loader),
                                    L_fd / len(train_loader),
                                    L_plasticity / len(train_loader),
                    train_acc,
            )    
            prog_bar.set_description(info)
        self.drift_estimator = copy.deepcopy(observer)
        logging.info(info)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def _difficulty_aware_kd(pred, soft, T, class_difficulty=None):
    """
    KD loss with class-specific difficulty attention mask.

    Args:
        pred: student logits, shape [N, C]
        soft: teacher logits, shape [N, C]
        T: temperature
        class_difficulty: None or 1D tensor/array length C. Larger->harder.

    Returns:
        scalar loss (averaged over batch)
    """
    log_preds = torch.log_softmax(pred / T, dim=1)
    soft_probs = torch.softmax(soft / T, dim=1)

    C = pred.shape[1]
    if class_difficulty is None:
        att = torch.ones(C, device=pred.device, dtype=pred.dtype)
    else:
        if not isinstance(class_difficulty, torch.Tensor):
            att = torch.tensor(class_difficulty, device=pred.device, dtype=pred.dtype)
        else:
            att = class_difficulty.to(pred.device).to(pred.dtype)

    # normalize attention to mean 1 to keep loss scale comparable
    att = att.unsqueeze(0)  # [1, C]

    weighted_soft = soft_probs * att
    loss = - (weighted_soft * log_preds).sum() / pred.shape[0]
    return loss


