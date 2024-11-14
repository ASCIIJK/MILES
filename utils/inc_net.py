import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from Backbones.Get_backbone import getbackbone
from Backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch.nn import functional as F
import scipy.stats as stats
import timm
import random


class AdopterBaseNet(nn.Module):
    def __init__(self, param):
        super(AdopterBaseNet, self).__init__()
        
        self.W_down = nn.ModuleList()
        self.W_up = nn.ModuleList()
        
        for i in range(len(param)-1):
            W_down = nn.Linear(param[i], param[i+1])
            W_down.weight.data = torch.zeros(param[i+1], param[i])
            W_down.bias.data = torch.zeros(param[i+1])
            self.W_down.append(W_down)
            
            W_up = nn.Linear(param[-1-i], param[-2-i])
            W_up.weight.data = torch.zeros(param[-2-i], param[-1-i])
            W_up.bias.data = torch.zeros(param[-2-i])
            self.W_up.append(W_up)

    def forward(self, x):
        
        x_inter = []
        
        for i in range(len(self.W_down)):
            x = F.relu(self.W_down[i](x))
            x_inter.append(x)
        
        for i in range(len(self.W_up)):
            if i == 0 or i == (len(self.W_up)-1):
                x = F.relu(self.W_up[i](x))
            else:
                x = F.relu(self.W_up[i](x + x_inter[-i-1]))

        return x


class PrototypeBase(nn.Module):
    def __init__(self, prototype):
        super().__init__()
        self.prototype = nn.Parameter(prototype, requires_grad=True)

    def freeze(self, mode=False):
        self.prototype.requires_grad = mode

    def freeze_adopter(self, mode=False):
        for p in self.adopter.parameters():
            p.requires_grad = mode

    def forward(self, hyper_features):
        # features: [bs, 512]
        features = hyper_features
        logits = torch.abs(features - self.prototype)
        logits = -torch.sum(logits, dim=1)
        return logits.unsqueeze(1)


class PrototypeLearnerV4(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        self.prototypes = nn.ModuleList()
        self.out_dim = out_dim
        self.adopter = nn.Linear(self.out_dim, self.out_dim)
        self.adopters = AdopterBaseNet(
            param=[self.out_dim, self.out_dim // 2, self.out_dim // 4, self.out_dim // 8, self.out_dim // 16])

        self.adopter.weight.data = torch.zeros(self.out_dim, self.out_dim)
        self.adopter.bias.data = torch.zeros(self.out_dim)

    def update_prototype(self, prototypes):
        for i in range(len(prototypes)):
            self.prototypes.append(PrototypeBase(prototypes[i]))

    def extract_feature(self, hyper_features):
        hyper_features = hyper_features.unsqueeze(0)
        with torch.no_grad():
            features = hyper_features
        return features

    def loss_plasticity_forward(self, hyper_features, targets, know_classes, masked=True):

        features = hyper_features + self.adopter(hyper_features) + self.adopters(hyper_features)

        prototypes = [self.prototypes[targets[i] - know_classes].prototype.unsqueeze(0) for i in range(features.shape[0])]
        prototypes = torch.concat(prototypes, dim=0)
        loss = torch.mean(torch.abs(features - prototypes), dim=1)

        if masked:
            sorted_tensor, _ = torch.sort(loss)
            num = sorted_tensor.shape[0] - sorted_tensor.shape[0] // 2
            threshold = sorted_tensor[num]
            loss[loss <= threshold] *= 0.

            loss2 = torch.exp(torch.sum(loss) / (sorted_tensor.shape[0] - num))
        else:
            loss2 = torch.exp(torch.mean(loss))
        
        return loss2

    def loss_stable_forward(self, hyper_features):
        features = hyper_features + self.adopter(hyper_features) + self.adopters(hyper_features)
        for i in range(len(self.prototypes)):
            if i == 0:
                dist = 1 / torch.exp(torch.mean(torch.abs(features - self.prototypes[i].prototype)))
            else:
                dist += 1 / torch.exp(torch.mean(torch.abs(features - self.prototypes[i].prototype)))

        return dist / len(self.prototypes)

    def freeze_self(self):
        for p in self.parameters():
            p.requires_grad = False

    def freeze_prototype(self, mode=False):
        for p in self.prototypes.parameters():
            p.requires_grad = mode

    def freeze_adopter(self, mode=False):
        for p in self.adopter.parameters():
            p.requires_grad = mode

    def forward(self, hyper_features):
        features = hyper_features + self.adopter(hyper_features) + self.adopters(hyper_features)
        logits = []
        for i in range(len(self.prototypes)):
            logits.append(self.prototypes[i](features))
        return logits
    
    def forward_run(self, hyper_features):
        logits = []
        for i in range(len(self.prototypes)):
            logits.append(self.prototypes[i](hyper_features))
        return logits


class MILESBase(nn.Module):
    def __init__(self, args):
        super(MILESBase, self).__init__()
        self.args = args
        self.backbone = getbackbone(args["backbone_type"])
        self.backbone.eval()
        self.prototype_table = nn.ModuleList()
        self.feature_dim = self.backbone.out_dim
        self.masked = args['distance_masked']
        self.cur_task = -1
        self.task_num = []

    def update_cat(self, num_classes):
        self.cur_task += 1
        self.task_num.append(num_classes)
        self.prototype_table.append(PrototypeLearnerV4(out_dim=self.feature_dim))

    def update_prototype(self, prototypes):
        self.prototype_table[-1].update_prototype(prototypes)

    def extract_features(self, x, targets=None):
        hyper_features = self.backbone(x)
        features = hyper_features
        return features

    def loss_self_supervised(self, inputs, targets):
        features = self.backbone(inputs)
        loss = self.prototype_table[-1].loss_plasticity_forward(features, targets, sum(self.task_num[:-1]), self.masked)

        return loss / self.task_num[-1]

    def loss_stable(self, hyper_features):
        return self.prototype_table[-1].loss_stable_forward(hyper_features)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = []
        for prototypeLearner in self.prototype_table:
            logits.extend(prototypeLearner(features))
        logits = torch.concat(logits, dim=1)
        return {"features": features,
                "logits": logits}
    
    def forward_run(self, x):
        features = self.backbone(x)
        logits = []
        for prototypeLearner in self.prototype_table:
            logits.extend(prototypeLearner.forward_run(features))
        logits = torch.concat(logits, dim=1)
        return {"features": features,
                "logits": logits}

    def forward_dist(self, features):
        logits = []
        for prototypeLearner in self.prototype_table:
            logits.extend(prototypeLearner(features))
        logits = torch.concat(logits, dim=1)
        return {"features": features,
                "logits": logits}
    
    
class PrototypeLearner(nn.Module):
    def __init__(self, prototype, masked=True, out_dim=768):
        super().__init__()
        self.prototype = nn.Parameter(prototype, requires_grad=True)
        self.out_dim = out_dim
        self.adapter = nn.Linear(self.out_dim, self.out_dim)
        self.adapter.weight.data = torch.zeros(self.out_dim, self.out_dim)
        self.adapter.bias.data = torch.zeros(self.out_dim)
        self.masked = masked

    def freeze_prototype(self, mode=False):
        self.prototype.requires_grad = mode

    def freeze_adapter(self, mode=False):
        for p in self.adapter.parameters():
            p.requires_grad = mode

    def loss_pls(self, hyper_features):
        features = hyper_features + self.adapter(hyper_features)
        prototypes = [self.prototype.unsqueeze(0) for i in range(features.shape[0])]
        prototypes = torch.concat(prototypes, dim=0)
        loss = torch.mean(torch.abs(features - prototypes), dim=1)
        loss = torch.exp(torch.mean(loss))
        return loss

    def loss_sta(self, hyper_features):
        features = hyper_features + self.adapter(hyper_features)
        loss = 1 / torch.exp(torch.mean(torch.abs(features - self.prototype)))
        return loss

    def forward(self, hyper_features):
        features = hyper_features + self.adapter(hyper_features)
        logits = torch.abs(features - self.prototype)
        logits = -torch.sum(logits, dim=1)
        return logits.unsqueeze(1)

    def forward_run(self, hyper_features):
        logits = torch.abs(hyper_features - self.prototype)
        logits = -torch.sum(logits, dim=1)
        return logits.unsqueeze(1)


class TaskLearner(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        self.class_learner = nn.ModuleList()
        self.task_num = 0
        self.class_num = None
        self.feature_dim = out_dim

    def update_class(self, prototypes, masked=True):
        self.class_num = len(prototypes)
        for i in range(self.class_num):
            class_learner = PrototypeLearner(prototypes[i], masked=masked, out_dim=self.feature_dim)
            self.class_learner.append(class_learner)

    def freeze_prototypes(self, mode=False):
        for i in range(self.class_num):
            self.class_learner[i].freeze_prototype(mode)

    def freeze_adapters(self, mode=False):
        for i in range(self.class_num):
            self.class_learner[i].freeze_adapter(mode)

    def loss_pls(self, hyper_features, index_num):
        return self.class_learner[index_num].loss_pls(hyper_features)

    def loss_sta(self, hyper_features, index_num):
        loss = self.class_learner[index_num].loss_sta(hyper_features)
        return loss

    def forward(self, hyper_features):
        logits = []
        for i in range(len(self.class_learner)):
            logits.append(self.class_learner[i](hyper_features))
        return logits

    def forward_run(self, hyper_features):
        logits = []
        for i in range(len(self.class_learner)):
            logits.append(self.class_learner[i].forward_run(hyper_features))
        return logits


class MILESPlusBackbone(nn.Module):
    def __init__(self, args):
        super(MILESPlusBackbone, self).__init__()
        self.args = args
        self.backbone = getbackbone(args["backbone_type"])
        self.masked = args["distance_masked"]
        self.backbone.eval()
        self.task_learners = nn.ModuleList()
        self.feature_dim = self.backbone.out_dim
        self.masked = args['distance_masked']
        self.cur_task = -1
        self.task_num = []

    def update_task(self, num_classes, prototypes):
        self.cur_task += 1
        self.task_num.append(num_classes)
        task_learner = TaskLearner(out_dim=self.feature_dim)
        task_learner.update_class(prototypes, self.masked)
        self.task_learners.append(task_learner)

    def loss_pls(self, x, index_num):
        # with torch.no_grad():
        hyper_features = self.backbone(x)
        loss = self.task_learners[-1].loss_pls(hyper_features, index_num)
        return loss

    def loss_sta(self, hyper_features, index_num):
        # hyper_features = self.backbone(x)
        loss = self.task_learners[-1].loss_sta(hyper_features, index_num)
        return loss

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def extract_features(self, x, targets=None):
        features = self.backbone(x)
        return features

    def forward(self, x):
        features = self.backbone(x)
        logits = []
        for task_learners in self.task_learners:
            logits.extend(task_learners(features))
        logits = torch.concat(logits, dim=1)
        return {"features": features,
                "logits": logits}

    def forward_run(self, x):
        hyper_features = self.backbone(x)
        logits = []
        for task_learners in self.task_learners:
            logits.extend(task_learners.forward_run(hyper_features))
        logits = torch.concat(logits, dim=1)
        return {"features": hyper_features,
                "logits": logits}

    def forward_feature(self, hyper_features):
        logits = []
        for task_learners in self.task_learners:
            logits.extend(task_learners.forward_run(hyper_features))
        logits = torch.concat(logits, dim=1)
        return {"features": hyper_features,
                "logits": logits}
