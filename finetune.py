import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
from datetime import datetime
import time
from tqdm import tqdm

class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CustomVGG19Model(torch.nn.Module):
    def __init__(self):
        super(CustomVGG19Model, self).__init__()

        model = models.vgg19(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CustomResNet50(torch.nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()

        model = models.resnet50(pretrained=True)
        model = models.googlenet(pretrained=True)
        self.features = model.fc.in_features
        # self.features = model.features

        for param in model.parameters():
            param.requires_grad = False

        # replace fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FilterPrunner(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset()
    
    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(torch.flatten(x, 1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(device)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v.cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune             

class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model) 
        self.model.train()

    def test(self):
        # return
        self.model.eval()
        output = None
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.to(device)
            label = label.to(device)
            with torch.no_grad():
                output = model(Variable(batch))
            # pred = output.data.max(1)[1]
            pred = output.argmax(1)
            correct += (pred == label).sum().item()
            # correct += pred.cpu().eq(label).sum()
            total += label.size(0)
        
        print("Accuracy :", float(correct) / total)
        
        self.model.train()

    def train(self, optimizer = None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for i in tqdm(range(epoches)):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")
        

    def train_batch(self, optimizer, batch, label, rank_filters):
        batch, label = batch.to(device), label.to(device)

        optimizer.zero_grad()

        if rank_filters:
            output = self.prunner(batch)
            loss = self.criterion(output, label)
        else:
            output = self.model(batch)
            loss = self.criterion(output, label)

        loss.backward()

        if not rank_filters:
            optimizer.step()

        return loss.item()

    def train_epoch(self, optimizer = None, rank_filters = False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, optimizer, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(optimizer=optimizer, rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)
        
    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        #Get the accuracy before prunning
        self.test()
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 3)

        print("Number of prunning iterations to reduce 67% filters", iterations)

        for _ in tqdm(range(iterations)):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(optimizer, num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")

            use_cuda = device.type == 'cuda'

            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=use_cuda)

            self.model = model
            self.model = self.model.to(device)

            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epoches = 10)


        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=15)

        current_time = datetime.now().strftime("%d_%B_%H:%M")
        torch.save(model, f"model_vg16_prunned_{current_time}")
        # torch.save(model.state_dict(), "model_prunned")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = None)
    parser.add_argument("--test_path", type = str, default = None)
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.add_argument("--model-path", type = str, default = None)
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda
    # args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args

if __name__ == '__main__':
    args = get_args()

    use_cuda = True
    if use_cuda:
        assert torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model_path
    if model_name is None:
        model_name = "model"

    if args.train:
        model = ModifiedVGG16Model()
        # model = CustomVGG19Model()
    elif args.prune:
        model = ModifiedVGG16Model()
        model.load_state_dict(torch.load(model_name, map_location=device))
        # model = torch.load(model_name, map_location=device, weights_only=False)

    model = model.to(device)

    # if args.use_cuda:
    #     model = model.cuda()

    train_path = None
    if args.train_path is None:
        train_path = "/home/washindeiru/studia/sem_8/ssn/sem/pytorch-pruning/data/animals10/train"
    else:
        train_path = args.train_path

    test_path = None
    if args.test_path is None:
        test_path = "/home/washindeiru/studia/sem_8/ssn/sem/pytorch-pruning/data/animals10/test"
    else:
        test_path = args.test_path


    fine_tuner = PrunningFineTuner_VGG16(train_path, test_path, model)

    if args.train:
        fine_tuner.train(epoches=10)
        current_time = datetime.now().strftime("%d_%B_%H:%M")
        torch.save(model.state_dict(), f"model_vg16_{current_time}.pth")
        # torch.save(model, "model")

    elif args.prune:
        fine_tuner.prune()
