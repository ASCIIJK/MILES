import math
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
from utils.inc_net import MILESPlusBackbone
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, count_parameters
import os

EPSILON = 1e-8


class MILES_Plus(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.num_workers = args["num_workers"]
        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_batch_size = args["init_batch_size"]
        self.batch_size = args["batch_size"]
        self.memory_size = args["memory_size"]
        self.memory_per_class = args["memory_per_class"]
        self.fixed = args["fixed_memory"]
        self.epochs = args["epochs"]
        self.weight_decay = args["weight_decay"]
        self.lr = args["lr"]
        self.pseudo_bs = args["pseudo_bs"]
        self._network = MILESPlusBackbone(args)

        self.init_class = args["init_cls"]
        self.increment = args["increment"]
        self.class_list = None
        self.DataManger = None
        self.task_num = []

        self._old_network = None
        self.device = args["device"]

        self.known_class = 0
        self.cur_task = -1
        self.task_acc = []
        self.TaskConfusionMatrix = []
        self.ConfusionMatrix = []
        self.pseudo_loader = None

        self.test_loader = None
        self.accs = {"all": [], "old": [], "new": []}
        self.logger = loger

        self.feature_mean = None
        self.feature_var = None

        self.prototype_table = None
        self.prototype_std_table = None

    def after_train(self):
        self._network.freeze()
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment

        acc, _, _ = self.eval_task(self.test_loader)
        self.accs["all"].append(round(acc, 2))
        self.logger.info("total acc: {}".format(self.accs["all"]))
        print("total acc: {}".format(self.accs["all"]))
        self.logger.info(self.TaskConfusionMatrix[-1])
        # self.logger.info(self.ConfusionMatrix[-1])

    def increment_train(self):
        self.cur_task += 1
        increment_class_list = self.class_list[self.known_class:self.known_class + self.increment]
        self.logger.info("training classes is {}".format(increment_class_list))

        train_dataset = self.DataManger.get_dataset(
            source="train", class_list=increment_class_list, appendent=None
        )
        train_dataset.labels = self.targets_map(train_dataset.labels)
        
        task_dataset = []
        for i in range(self.increment):
            train_dataset = self.DataManger.get_dataset(
                source="train", class_list=[increment_class_list[i]], appendent=None
            )
            train_dataset.labels = self.targets_map(train_dataset.labels)
            task_dataset.append(train_dataset)

        test_dataset = self.DataManger.get_dataset(
            source="test", class_list=self.class_list[:self.known_class + self.increment], appendent=None
        )
        self.test_dataset = test_dataset
        test_dataset.labels = self.targets_map(test_dataset.labels)
        # 构建迭代器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        
        task_loader = []
        for i in range(self.increment):
            train_dataset = task_dataset[i]
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
            )
            task_loader.append(train_loader)
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        self.test_loader = test_loader

        for p in self._network.backbone.parameters():
            p.requires_grad = False

        prototype_set = self.DataManger.get_dataset(
            source="prototype", class_list=increment_class_list, appendent=None
        )
        prototype_set.labels = self.targets_map(prototype_set.labels)
        prototype_loader = DataLoader(
            prototype_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers
        )

        new_prototype, prototype_stds = self.get_covmatrix(prototype_loader, increment_class_list)

        self.prototype_table.extend(new_prototype)
        self.prototype_std_table.extend(prototype_stds)

        self.pseudo_loader = self.DataManger.generate_pseudo_features(self.prototype_table[:-self.increment],
                                                                      self.prototype_std_table[:-self.increment],
                                                                      range(self.known_class), 512, bs=self.pseudo_bs)

        self._network.update_task(self.increment, new_prototype)

        self._train_self_supervised(task_loader, test_loader)

    def init_train(self, datamanger):
        self.DataManger = datamanger
        self.class_list = datamanger.class_order
        self.logger.info("init_class: {}, increment: {}".format(self.init_class, self.increment))

        self.cur_task += 1
        init_class_list = self.class_list[0:self.init_class]

        self.logger.info("training classes is {}".format(init_class_list))

        train_dataset = self.DataManger.get_dataset(
            source="train", class_list=init_class_list, appendent=None
        )
        train_dataset.labels = self.targets_map(train_dataset.labels)

        task_dataset = []
        for i in range(self.init_class):
            train_dataset = self.DataManger.get_dataset(
                source="train", class_list=[init_class_list[i]], appendent=None
            )
            train_dataset.labels = self.targets_map(train_dataset.labels)
            task_dataset.append(train_dataset)

        test_dataset = self.DataManger.get_dataset(
            source="test", class_list=self.class_list[:self.init_class], appendent=None
        )
        self.test_dataset = test_dataset
        test_dataset.labels = self.targets_map(test_dataset.labels)

        task_loader = []
        for i in range(self.init_class):
            train_dataset = task_dataset[i]
            train_loader = DataLoader(
                train_dataset, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers
            )
            task_loader.append(train_loader)

        test_loader = DataLoader(
            test_dataset, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers
        )
        self.test_loader = test_loader

        for p in self._network.backbone.parameters():
            p.requires_grad = False

        prototype_set = self.DataManger.get_dataset(
            source="prototype", class_list=init_class_list, appendent=None
        )
        prototype_set.labels = self.targets_map(prototype_set.labels)
        prototype_loader = DataLoader(
            prototype_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers
        )

        new_prototype, prototype_stds = self.get_covmatrix(prototype_loader, init_class_list)

        self.prototype_table = new_prototype
        self.prototype_std_table = prototype_stds

        self._network.update_task(self.init_class, new_prototype)

        self._train_self_supervised(task_loader, test_loader)

    def _run(self, train_loader, test_loader):
        self._network.to(self.device)

        for i in range(self.cur_task + 1):
            if i < self.cur_task:
                self._network.task_learners[i].freeze_prototypes()
            else:
                self._network.task_learners[i].freeze_prototypes(mode=True)

        for i in range(self.cur_task + 1):
            self._network.task_learners[i].freeze_adapters()

        epochs = self.args['center_correct_epochs']
        lr = self.args['center_correct_lr']
        weight_decay = self.args['center_correct_decay']

        prog_bar = tqdm(range(epochs))

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs
        )

        if self.cur_task > 0:
            loader = iter(self.pseudo_loader)

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self._network.forward_run(inputs)
                logits = outputs["logits"]

                if self.cur_task > 0:
                    try:
                        _, features, label = next(loader)
                    except StopIteration:
                        loader = iter(self.pseudo_loader)
                        _, features, label = next(loader)
                    features, label = features.to(self.device), label.to(self.device)
                    logits2 = self._network.forward_feature(features)["logits"]
                    logits = torch.cat((logits, logits2), dim=0)
                    targets = torch.cat((targets, label), dim=0)

                loss = F.cross_entropy(logits, targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self.compute_test_accuracy(test_loader, self._network, mode=False)

            if self.cur_task == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self.cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self.cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc
                )
            self.logger.info(info)
            prog_bar.set_description(info)

    def _train_self_supervised(self, task_loader, test_loader):
        self._network = self._network.to(self.device)

        for i in range(self.cur_task + 1):
            self._network.task_learners[i].freeze_prototypes()

        for i in range(self.cur_task + 1):
            if i < self.cur_task:
                self._network.task_learners[i].freeze_adapters()
            else:
                self._network.task_learners[i].freeze_adapters(mode=True)

        self.logger.info("All params: {}".format(count_parameters(self._network)))
        self.logger.info("Trainable params: {}".format(count_parameters(self._network, True)))

        if self.cur_task > 0:
            epochs = self.args['epochs']
            lr = self.args['lr']
            weight_decay = self.args['weight_decay']
        else:
            epochs = self.args['init_epochs']
            lr = self.args['init_lr']
            weight_decay = self.args['init_weight_decay']

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs
        )

        prog_bar = tqdm(range(epochs))

        if self.cur_task > 0:
            loader = iter(self.pseudo_loader)
        else:
            loader = None

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i in range(self._network.task_num[-1]):
                losses_c, total = self.run_class(optimizer, task_loader[i], loader, i)
                losses += losses_c
            scheduler.step()

            info = "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                self.cur_task,
                epoch + 1,
                epochs,
                losses / total,
            )
            self.logger.info(info)
            prog_bar.set_description(info)

    def run_class(self, optimizer, train_loader, loader, index_num):
        correct, total = 0, 0
        losses = 0.0
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            loss = self._network.loss_pls(inputs, index_num) * 1.0

            if self.cur_task > 0:
                try:
                    _, features, label = next(loader)
                except StopIteration:
                    loader = iter(self.pseudo_loader)
                    _, features, label = next(loader)
                features, label = features.to(self.device), label.to(self.device)
                loss_sta = 0.05 * self._network.loss_sta(features, index_num)
                loss = loss + loss_sta

            total += len(targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        return losses, total

    def update_prototype(self, loader, class_list):
        new_prototype = torch.zeros(len(class_list), self._network.backbone.out_dim)
        targets_num = torch.zeros(len(class_list), 1)
        prog_bar = tqdm(loader)
        model = self._network
        model = model.to(self.device)
        print('update the prototype')
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                features = model.extract_features(inputs)
            for j in range(len(targets)):
                new_prototype[targets[j] - self.known_class, :] += features[j, :].cpu()
                targets_num[targets[j] - self.known_class] += 1
        new_prototype /= targets_num

        prototype_std = torch.zeros(len(class_list), self._network.backbone.out_dim)
        targets_num = torch.zeros(len(class_list), 1)
        prog_bar = tqdm(loader)
        print('update the prototype_std')
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                features = model.extract_features(inputs)
            for j in range(len(targets)):
                prototype_std[targets[j] - self.known_class, :] += (features[j, :].cpu() - new_prototype[targets[j] - self.known_class, :])**2
                targets_num[targets[j] - self.known_class] += 1
        prototype_std /= targets_num
        prototype_std = torch.sqrt(prototype_std)

        if self.prototype_table is None:
            self.prototype_table = new_prototype
        else:
            self.prototype_table = torch.cat([self.prototype_table, new_prototype], dim=0)
        return new_prototype, prototype_std

    def get_avgdistance_disstd(self, loader, class_list, new_prototype):
        avg_dis = torch.zeros(len(class_list))
        std_dis = torch.zeros(len(class_list))
        model = self._network
        model = model.to(self.device)
        for i in range(len(class_list)):
            dis = []
            for ii, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    features = model.extract_features(inputs)
                for j in range(len(targets)):
                    if targets[j] == class_list[i]:
                        dis.append(torch.sum(torch.abs(features[j, :].cpu() - new_prototype[i])).unsqueeze(0))
            dis = torch.concat(dis, dim=0)
            avg_dis[i] = torch.mean(dis)
            std_dis[i] = torch.std(dis)
        return avg_dis, std_dis

    def get_another_protype(self, newprototype, loader, class_list):
        new_prototype1 = torch.zeros(len(class_list), self._network.backbone.out_dim).cuda()
        targets_num1 = torch.zeros(len(class_list), 1)
        model = self._network
        model = model.to(self.device)
        newprototype = newprototype.cuda()
        print('get the max prototype from center')
        prog_bar = tqdm(loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                features = model.extract_features(inputs)
            for j in range(len(targets)):
                distance = torch.sum(torch.abs(features[j, :] - newprototype[targets[j] - self.known_class, :]))
                if distance.cpu() > targets_num1[targets[j] - self.known_class]:
                    new_prototype1[targets[j] - self.known_class, :] = features[j, :]
                    targets_num1[targets[j] - self.known_class] = distance

        new_prototype2 = torch.zeros(len(class_list), self._network.backbone.out_dim).cuda()
        targets_num2 = torch.zeros(len(class_list), 1)

        print('get the max prototype from last prototype')
        prog_bar = tqdm(loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                features = model.extract_features(inputs)
            for j in range(len(targets)):
                distance = torch.sum(torch.abs(features[j, :] - new_prototype1[targets[j] - self.known_class, :]))
                if distance.cpu() > targets_num2[targets[j] - self.known_class]:
                    new_prototype2[targets[j] - self.known_class, :] = features[j, :]
                    targets_num2[targets[j] - self.known_class] = distance

        return new_prototype1, new_prototype2

    def get_max_similarity(self, loader, class_list):
        # in_features: [bs, 197, 768]
        all_features = []
        for i in class_list:
            all_features.append([])
        model = self._network
        model = model.to(self.device)
        print('+++++++++++get the features+++++++++++++')
        prog_bar = tqdm(loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                feature = model.extract_features(inputs, targets)
                for j in range(feature.shape[0]):
                    all_features[targets[j] - self.known_class].append(feature[j, :].unsqueeze(0))
        for i in range(len(class_list)):
            all_features[i] = torch.concat(all_features[i], dim=0)

        prototypes, proto_nums = [], []
        dists = []
        for i in range(len(class_list)):
            prototype = torch.mean(all_features[i], dim=0).unsqueeze(0)  # [1, 197, 768]
            dist = all_features[i] * prototype  # [n, 197, 768]
            dist = torch.sum(dist, dim=2)  # [n, 197]
            dist = dist / torch.sum(all_features[i] ** 2, dim=2) / torch.sum(prototype ** 2, dim=2)  # [n, 197]
            dist = torch.mean(dist, dim=0)
            distss, min_num = torch.max(dist, dim=0)
            dists.append(distss)
            prototypes.append(prototype[:, min_num, :].squeeze())
            proto_nums.append(min_num)
        print(proto_nums)
        print(dists)
        return prototypes, proto_nums

    def get_covmatrix(self, loader, class_list):
        all_features = []
        for i in class_list:
            all_features.append([])
        model = self._network
        model = model.to(self.device)
        print('get the features')
        prog_bar = tqdm(loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                feature = model.extract_features(inputs, targets)
                for j in range(feature.shape[0]):
                    all_features[targets[j] - self.known_class].append(feature[j, :].unsqueeze(0))
        for i in range(len(class_list)):
            all_features[i] = torch.concat(all_features[i], dim=0)

        prototypes = []
        prototype_stds = []

        for i in range(len(class_list)):
            # Calculate prototype
            features = all_features[i]
            prototype = torch.mean(features, dim=0)
            prototypes.append(prototype)
            prototype_std = torch.std(features, dim=0)
            prototype_stds.append(prototype_std)
        return prototypes, prototype_stds

    def compute_loss(self, features, targets):
        logits = []
        for i in range(len(targets)):
            logits.append(self.compute_distance(features[i, :]).unsqueeze(0))
        logits = torch.concat(logits, dim=0)
        loss = F.cross_entropy(logits, targets.long())
        return loss

    def compute_distance(self, feature):
        prototype_table = self.prototype_table.to(self.device)
        distances = torch.norm(prototype_table - feature, dim=1)
        return -distances

    def prototype_normalize(self, prototype):
        row_min = prototype.min(dim=1, keepdim=True)[0]
        row_max = prototype.max(dim=1, keepdim=True)[0]

        return (prototype - row_min) / (row_max - row_min + 1e-8)

    def targets_map(self, targets):
        for i in range(len(targets)):
            targets[i] = self.class_list.index(targets[i])
        return targets

    def get_results(self, features):
        min_index = []
        for i in range(features.shape[0]):
            min_index.append(self.apply2each_line(features[i, :]))
        return torch.tensor(min_index)

    def get_train_results(self, features):
        min_index = []
        for i in range(features.shape[0]):
            min_index.append(self.apply2each_train_line(features[i, :]))
        return torch.tensor(min_index)

    def apply2each_line(self, feature):
        prototype_table = self.prototype_table.to(self.device)
        distances = torch.norm(feature - prototype_table, dim=1)
        min_index = torch.argmin(distances)
        return min_index

    def compute_test_accuracy(self, test_loader, model, mode=None):
        model.eval()
        correct, total = 0, 0
        device = self.device
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                if mode is None:
                    outputs = model(inputs)
                else:
                    outputs = model.forward_run(inputs)
            logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def eval_task(self, test_loader):
        model = self._network.to(self.device)
        model.eval()
        confusion_matrix = np.zeros((self.known_class, self.known_class), dtype=int)
        task_confusion_matrix = np.zeros((self.cur_task + 1, self.cur_task + 1), dtype=int)
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)
            logits = outputs["logits"]
            predicts = torch.max(logits, dim=1)[1]
            for j in range(len(predicts)):
                if predicts[j] < self.init_class:
                    pred_tsak = 0
                else:
                    pred_tsak = int(predicts[j] - self.init_class) // int(self.increment) + 1
                if targets[j] < self.init_class:
                    true_task = 0
                else:
                    true_task = int(targets[j] - self.init_class) // int(self.increment) + 1
                task_confusion_matrix[true_task, pred_tsak] += 1
                confusion_matrix[targets[j], predicts[j]] += 1
        correct = 0.
        for i in range(self.known_class):
            correct += confusion_matrix[i, i]
        acc = correct / confusion_matrix.sum()

        if self.cur_task == 0:
            old_task = []
            new_task = range(self.known_class)
        elif self.cur_task == 1:
            old_task = range(self.init_class)
            new_task = range(self.init_class, self.known_class)
        else:
            old_task = range(self.known_class-self.increment)
            new_task = range(self.known_class-self.increment, self.known_class)
        correct = 0.
        total = 0.
        for i in old_task:
            correct += confusion_matrix[i, i]
            total += confusion_matrix[i, :].sum()
        old_acc = 0. if total == 0. else correct / total
        correct = 0.
        total = 0.
        for i in new_task:
            correct += confusion_matrix[i, i]
            total += confusion_matrix[i, :].sum()
        new_acc = 0. if total == 0. else correct / total
        self.TaskConfusionMatrix.append(task_confusion_matrix)
        self.ConfusionMatrix.append(confusion_matrix)
        model.train()
        return acc * 100., old_acc * 100., new_acc * 100.
