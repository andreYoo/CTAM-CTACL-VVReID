from __future__ import print_function

import torch
import torch.nn as nn
import pdb
import numpy as np
import torch.nn.functional as F

device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')

class CTAM_SSCL_Loss(nn.Module):
    """A loss functions for the: camera-tracklet-awareness memory-based Semi-supervised contrastive learning"""
    def __init__(self, temperature=0.07, contrast_mode='all',base_temperature=0.07):
        super(CTAM_SSCL_Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.global_select = 5
        self.grey_zone_rate = 0.01
        self.base_temperature = base_temperature

    def forward(self,memory,logits,camids,hard_pos=None,trackids=None,type='local',thr=0.5):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if type=='local':
            loss = []
            _hard_pos = []
            for i,feat in enumerate(logits):
                _cam_logit = feat[memory.mem_CID==camids[i]] #basement
                _tid_list = memory.mem_TID[memory.mem_CID==camids[i]]
                _hard_sim_index = torch.argmin(_cam_logit[_tid_list==trackids[i]])
                _th_pos = memory.mem[memory.mem_CID==camids[i]][_tid_list==trackids[i]][_hard_sim_index]
                if len(np.shape(_th_pos))>=2: # if there are more than two features are detected
                    _features = memory.mem[memory.mem_CID==camids[i]]
                    _features = _features[_tid_list==trackids[i]]
                    _centre_feature = torch.mean(_features,0)
                    _dist = torch.cdist(torch.unsqueeze(_centre_feature,0),_th_pos)
                    _dist_index = torch.argmax(_dist)
                    _th_pos  = _th_pos[_dist_index]
                _hard_pos.append(_th_pos)

                anchor_dot_contrast = torch.div(_cam_logit,self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=0, keepdim=True)
                anchor_dot_contrast = anchor_dot_contrast - logits_max.detach() #Stabilisation

                _positive = anchor_dot_contrast[_tid_list==trackids[i]] #positive
                _num_positive = len(_positive)

                exp_logits = torch.exp(anchor_dot_contrast)
                # mask-out self-contrast cases
                log_prob = _positive - torch.log(exp_logits.sum(0, keepdim=True))

                # compute mean of log-likelihood over positive
                mean_log_prob_pos = (log_prob).sum(0) / (_num_positive)
                mean_log_prob_pos = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss.append(mean_log_prob_pos)

            return torch.mean(torch.stack(loss)),torch.stack(_hard_pos)

        elif type=='global':
            loss = []
            pos_list =  []
            _hard_pos_sim = torch.squeeze(hard_pos).mm(memory.mem.t())
            for i,feat in enumerate(logits):
                _cam_logit = feat[memory.mem_CID!=camids[i]] #basement
                _cam_hard_pos_logits = _hard_pos_sim[i,memory.mem_CID!=camids[i]]
                _t1, _hard_pos_idx = torch.sort(_cam_hard_pos_logits.detach().clone(), descending=True)
                _t2, _easy_pos_idx = torch.sort(_cam_logit.detach().clone(), descending=True)

                num = int(self.grey_zone_rate * _cam_logit.size(0))
                anchor_dot_contrast = torch.div(_cam_logit,self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=0, keepdim=True)
                anchor_dot_contrast = anchor_dot_contrast - logits_max.detach() #Stabilisation
                
                _hard_positive = anchor_dot_contrast[_hard_pos_idx[0:self.global_select]]
                _easy_positive = anchor_dot_contrast[_easy_pos_idx[0:self.global_select]]
                _num_positive = len(_hard_positive)+len(_easy_positive)

                if _num_positive==0:
                    _positive = torch.max(logits)
                    _num_positive = 1.0
                pos_list.append(_num_positive)



                _negative = anchor_dot_contrast[_hard_pos_idx[num:]]
                _positive = torch.cat([_hard_positive,_easy_positive])
                exp_logits = torch.exp(torch.cat([_positive,_negative]))

                #exp_logits = torch.exp(anchor_dot_contrast)
                # mask-out self-contrast cases

                log_prob = _positive - torch.log(exp_logits.sum(0, keepdim=True))

                # compute mean of log-likelihood over positive
                mean_log_prob_pos = (log_prob).sum(0) / (_num_positive)
                mean_log_prob_pos = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss.append(mean_log_prob_pos)
            return torch.mean(torch.stack(loss)),np.mean(pos_list)

        if type=='cl':
            loss = []
            _hard_pos = []
            for i,feat in enumerate(logits):
                _cam_logit = feat[memory.mem_CID==camids[i]] #basement
                _tid_list = memory.mem_TID[memory.mem_CID==camids[i]]
                _hard_sim_index = torch.argmin(_cam_logit[_tid_list==trackids[i]])
                _th_pos = memory.mem[memory.mem_CID==camids[i]][_tid_list==trackids[i]][_hard_sim_index]
                if len(np.shape(_th_pos))>=2: # if there are more than two features are detected
                    _features = memory.mem[memory.mem_CID==camids[i]]
                    _features = _features[_tid_list==trackids[i]]
                    _centre_feature = torch.mean(_features,0)
                    _dist = torch.cdist(torch.unsqueeze(_centre_feature,0),_th_pos)
                    _dist_index = torch.argmax(_dist)
                    _th_pos  = _th_pos[_dist_index]
                _hard_pos.append(_th_pos)

                anchor_dot_contrast = torch.div(_cam_logit,self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=0, keepdim=True)
                anchor_dot_contrast = anchor_dot_contrast - logits_max.detach() #Stabilisation

                _positive = anchor_dot_contrast[_tid_list==trackids[i]] #positive of intra cam
                _num_positive = len(_positive)
                #exp_logits = torch.exp(anchor_dot_contrast)  #negative of intra cam


                #Inter-cam-----------------------------------------------------------------------------------------------
                _hard_pos_sim = torch.unsqueeze(_th_pos,0).mm(memory.mem.t())
                _inter_cam_logit = feat[memory.mem_CID != camids[i]]  # basement
                _inter_cam_hard_pos_logits = torch.squeeze(_hard_pos_sim[:,memory.mem_CID != camids[i]])

                _t1, _hard_pos_idx = torch.sort(_inter_cam_hard_pos_logits.detach().clone(), descending=True)
                _t2, _easy_pos_idx = torch.sort(_inter_cam_logit.detach().clone(), descending=True)

                _greyz_num = int(self.grey_zone_rate * _inter_cam_logit.size(0))
                _inter_anchor_dot_contrast = torch.div(_inter_cam_logit, self.temperature)
                _intra_logits_max, _ = torch.max(_inter_anchor_dot_contrast, dim=0, keepdim=True)
                _inter_anchor_dot_contrast = _inter_anchor_dot_contrast - logits_max.detach()  # Stabilisation

                _hard_positive = _inter_anchor_dot_contrast[_hard_pos_idx[0:self.global_select]]
                _easy_positive = _inter_anchor_dot_contrast[_easy_pos_idx[0:self.global_select]]
                _intra_num_positive = len(_hard_positive) + len(_easy_positive)

                _inter_negative = _inter_anchor_dot_contrast[_hard_pos_idx[_greyz_num:]]
                _inter_positive = torch.cat([_hard_positive, _easy_positive])
                _inter_denominator = torch.cat([_inter_positive, _inter_negative])
                # Inter-cam-----------------------------------------------------------------------------------------------

                #Aggregation
                _neumorator = torch.cat([_positive,_inter_positive])
                _demonimator = torch.cat([anchor_dot_contrast,_inter_denominator])


                exp_logits = torch.exp(_demonimator)

                # mask-out self-contrast cases
                log_prob = _neumorator - torch.log(exp_logits.sum(0, keepdim=True))

                # compute mean of log-likelihood over positive
                _num_positive= _num_positive+_intra_num_positive
                mean_log_prob_pos = (log_prob).sum(0) / (_num_positive)
                mean_log_prob_pos = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss.append(mean_log_prob_pos)
            return torch.mean(torch.stack(loss)),torch.stack(_hard_pos)

        elif type=='cam':
            valid_cam_nums, valid_cam_ids = memory.set_cam_memory()
            uniform_dist = torch.Tensor(logits.size(0),valid_cam_nums).fill_((1./valid_cam_nums)).to(device_1)
            cam_likelihood = F.log_softmax(logits.mm(memory.mem_cam[valid_cam_ids].t()),dim=1)
            if valid_cam_nums != memory.num_cam:
                return F.kl_div(cam_likelihood,uniform_dist)*valid_cam_nums
            else:
                return F.kl_div(cam_likelihood,uniform_dist)*memory.num_cam



