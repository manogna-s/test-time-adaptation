import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import wandb
import numpy as np

from copy import deepcopy
from models.model import ResNetDomainNet126


logger = logging.getLogger(__name__)


class ACL_TTAMethod(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.episodic = cfg.MODEL.EPISODIC
        self.dataset_name = cfg.CORRUPTION.DATASET
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"

        # configure model and optimizer
        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.print_amount_trainable_params()

        # variables needed for single sample test-time adaptation (sstta) using a sliding window (buffer) approach
        self.input_buffer = None
        self.window_length = cfg.TEST.WINDOW_LENGTH
        self.pointer = torch.tensor([0], dtype=torch.long).cuda()
        # sstta: if the model has no batchnorm layers, we do not need to forward the whole buffer when not performing any updates
        self.has_bn = any(
            [isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()])

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # active learning parameters
        self.num_active_samples = cfg.ACL.NUM_ACTIVE_SAMPLES
        self.acl_strategy = cfg.ACL.STRATEGY

        # source like feature bank for active learning
        self.acl_bank = {
            "features": torch.tensor([], device="cuda"),
            "probs": torch.tensor([], device="cuda"),
            "labels": torch.tensor([], device="cuda", dtype=torch.long)
        }

        # save batch wise analysis
        self.n_correct = {'act': 0, 'others': 0, 'all': 0, 'act_nn': 0}
        self.n_samples = {'act': 0, 'others': 0, 'all': 0, 'act_nn': 0}

    def forward(self, x, y):
        if self.episodic:
            self.reset()

        x = x if isinstance(x, list) else [x]

        if x[0].shape[0] == 1:  # single sample test-time adaptation
            # create the sliding window input
            if self.input_buffer is None:
                self.input_buffer = [x_item for x_item in x]
                # set bn1d layers into eval mode, since no statistics can be extracted from 1 sample
                self.change_mode_of_batchnorm1d(
                    self.models, to_train_mode=False)
            elif self.input_buffer[0].shape[0] < self.window_length:
                self.input_buffer = [
                    torch.cat([self.input_buffer[i], x_item], dim=0) for i, x_item in enumerate(x)]
                # set bn1d layers into train mode
                self.change_mode_of_batchnorm1d(
                    self.models, to_train_mode=True)
            else:
                for i, x_item in enumerate(x):
                    self.input_buffer[i][self.pointer] = x_item

            if self.pointer == (self.window_length - 1):
                # update the model, since the complete buffer has changed
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(self.input_buffer)
                outputs = outputs[self.pointer.long()]
            else:
                # create the prediction without updating the model
                if self.has_bn:
                    # forward the whole buffer to get good batchnorm statistics
                    outputs = self.forward_sliding_window(self.input_buffer)
                    outputs = outputs[self.pointer.long()]
                else:
                    # only forward the current test sample, since there are no batchnorm layers
                    outputs = self.forward_sliding_window(x)

            # increase the pointer
            self.pointer += 1
            self.pointer %= self.window_length

        else:   # common batch adaptation setting
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x, y)

        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        """
        raise NotImplementedError

    def active_sample_selection(self, feats_w, logits_w, probs_w):
        if self.acl_strategy == "random":
            acl_idx = torch.randint(
                0, feats_w.shape[0], (self.num_active_samples,))

        elif self.acl_strategy == "entropy":
            ent = - torch.sum(probs_w * torch.log(probs_w + 1e-6), dim=1)
            ent_sorted, idx_sorted = torch.sort(ent, descending=True)
            acl_idx = idx_sorted[:self.num_active_samples]

        elif self.acl_strategy == "mhpl":

            # acl nn using feature bank
            cos_sim = torch.matmul(F.normalize(feats_w, dim=1), F.normalize(
                self.acl_bank["features"], dim=1).T)
            cos_sim_sorted, idxs = cos_sim.sort(descending=True)
            cos_sim_sorted, idxs = cos_sim_sorted[:, 1:], idxs[:, 1:]

            idxs = idxs[:, :self.num_neighbors]
            na = (cos_sim_sorted[:, :self.num_neighbors]).mean(1)

            probs = self.acl_bank["probs"]
            _, bank_preds = torch.max(probs, dim=1)
            preds_nn = bank_preds[idxs]

            # num_classes = probs.shape[1]

            pred_dist_nn = torch.zeros_like(logits_w)
            for c in range(probs.shape[1]):
                pred_dist_nn[:, c] = torch.sum(preds_nn == c, dim=1)
            pred_dist_nn = pred_dist_nn/self.num_neighbors

            nnp = - torch.sum(pred_dist_nn *
                                torch.log(pred_dist_nn + 1e-6), dim=1)
            
            ent = - torch.sum(probs_w * torch.log(probs_w + 1e-6), dim=1)

            neu = na * ent # nnp * na
            neu_sorted, idx_sorted = torch.sort(neu, descending=True)
            acl_idx = idx_sorted[:self.num_active_samples]

        elif self.acl_strategy == "ours":
            if self.acl_bank["features"].shape[0] == 0:
                acl_idx = torch.randint(
                    0, y.shape[0], (self.num_active_samples,))
                cos_sim = torch.matmul(F.normalize(feats_w, dim=1), F.normalize(
                    feats_w, dim=1).T)
                cos_sim_sorted, idxs = cos_sim.sort(descending=True)
                cos_sim_sorted, idxs = cos_sim_sorted[:, 1:], idxs[:, 1:]

                bank_preds = logits_w.argmax(1)

                idxs = idxs[:, : self.num_neighbors]
            else:
                cos_sim = torch.matmul(F.normalize(
                    feats_w, dim=1), F.normalize(feats_w, dim=1).T)
                cos_sim_sorted, idxs = cos_sim.sort(descending=True)
                cos_sim_sorted, idxs = cos_sim_sorted[:, 1:], idxs[:, 1:]

                idxs = idxs[:, :self.num_neighbors]
                na = (cos_sim_sorted[:, :self.num_neighbors]).mean(1)

                probs = logits_w.softmax(1)
                _, sample_preds = torch.max(probs, dim=1)
                preds_nn = sample_preds[idxs]

                num_classes = probs.shape[1]

                pred_dist_nn = torch.zeros_like(logits_w)
                for c in range(num_classes):
                    pred_dist_nn[:, c] = torch.sum(preds_nn == c, dim=1)
                pred_dist_nn = pred_dist_nn/self.num_neighbors

                # nnp = - torch.sum(pred_dist_nn * torch.log(pred_dist_nn + 1e-6), dim=1)

                nnp = torch.sum(sample_preds.unsqueeze(1) ==
                                preds_nn, dim=1)/self.num_neighbors

                nnp = (np.log(self.num_neighbors) + torch.sum(pred_dist_nn *
                       torch.log(pred_dist_nn + 1e-6), dim=1)) / np.log(self.num_neighbors)

                ent = -torch.sum(probs * torch.log(probs+1e-6),
                                 dim=1) / np.log(num_classes)

                neu = nnp * ent * na
                neu_sorted, idx_sorted = torch.sort(neu, descending=True)
                acl_idx = idx_sorted[:self.num_active_samples]

        cos_sim_batch = torch.matmul(F.normalize(
            feats_w, dim=1), F.normalize(feats_w, dim=1).T)
        cos_sim_batch_sorted, idxs_batch = cos_sim_batch.sort(descending=True)
        cos_sim_batch_sorted, idxs_batch = cos_sim_batch_sorted[:,
                                                                1:], idxs_batch[:, 1:]
        idxs_batch = idxs_batch[:, : self.num_neighbors]
        acl_nn_batch = idxs_batch[acl_idx]

        return acl_idx, acl_nn_batch

    @torch.no_grad()
    def post_tta_analysis(self, images_test, y, acl_idx, acl_nn_batch):
        self.model.eval()
        _, logits = self.model(images_test, cls_only=True)
        post_preds = logits.argmax(1)
        post_batch_acc = (post_preds == y).float().sum()

        post_act = post_preds[acl_idx]
        post_act_nn_batch = post_preds[acl_nn_batch]
        post_act_acc = (post_act == y[acl_idx]).float()
        post_act_nn_batch_acc = (post_act_nn_batch == y[acl_nn_batch]).float()

        self.n_correct['act'] += post_act_acc.sum().cpu().item()
        self.n_correct['others'] += post_batch_acc.sum().cpu().item() - \
            post_act_acc.sum().cpu().item()
        self.n_correct['all'] += post_batch_acc.sum().cpu().item()
        self.n_correct['act_nn'] += post_act_nn_batch_acc.sum().cpu().item()

        self.n_samples['act'] += self.num_active_samples
        self.n_samples['others'] += images_test.shape[0] - \
            self.num_active_samples
        self.n_samples['all'] += images_test.shape[0]

        batch_acl_dict = {'act_correct': self.n_correct['act'], 'others_correct': self.n_correct['others'],
                          'all_correct': self.n_correct['all'], 'act_nn_correct': self.n_correct['act_nn']}

        batch_acc_dict = {'act': self.n_correct['act']/self.n_samples['act'], 'others': self.n_correct['others'] /
                          self.n_samples['others'], 'all': self.n_correct['all']/self.n_samples['all']}
        wandb.log(batch_acc_dict)
        return

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model(imgs_test)

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        if self.cfg.OPTIM.METHOD == 'Adam':
            return torch.optim.Adam(self.params,
                                    lr=self.cfg.OPTIM.LR,
                                    betas=(self.cfg.OPTIM.BETA, 0.999),
                                    weight_decay=self.cfg.OPTIM.WD)
        elif self.cfg.OPTIM.METHOD == 'SGD':
            return torch.optim.SGD(self.params,
                                   lr=self.cfg.OPTIM.LR,
                                   momentum=self.cfg.OPTIM.MOMENTUM,
                                   dampening=self.cfg.OPTIM.DAMPENING,
                                   weight_decay=self.cfg.OPTIM.WD,
                                   nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    def print_amount_trainable_params(self):
        trainable = sum(p.numel()
                        for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"#Trainable/total parameters: {trainable}/{total} \t Fraction: {trainable / total * 100:.2f}% ")

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    @staticmethod
    def copy_model(model):
        # https://github.com/pytorch/pytorch/issues/28594
        if isinstance(model, ResNetDomainNet126):
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        delattr(module, hook.name)
            coppied_model = deepcopy(model)
            for module in model.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
        else:
            coppied_model = deepcopy(model)
        return coppied_model

    @staticmethod
    def change_mode_of_batchnorm1d(model_list, to_train_mode=True):
        # batchnorm1d layers do not work with single sample inputs
        for model in model_list:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    if to_train_mode:
                        m.train()
                    else:
                        m.eval()
