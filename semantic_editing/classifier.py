import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
import sys

import torchvision.models as models

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, input_nc, pool_nd, _nclass, _lr=0.001, _beta1=0.5, _nepoch=5, _batch_size=1):
        self.train_X =  _train_X 
        self.train_Y = _train_Y 
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.pool_nd = pool_nd
        self.input_nc = input_nc

        # self.input_dim = _input_dim
        # self.cuda = _cuda
        self.model = NLayerDiscriminator_shape(self.input_nc, pool_dim=self.pool_nd, num_class = self.nclass) #self.nclass
        # self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        
        # self.input = torch.FloatTensor(_batch_size, self.input_dim)
        # self.label = torch.LongTensor(_batch_size)
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=_lr, betas=(_beta1, 0.999))


        self.model.cuda()
        self.criterion.cuda()
            # self.input = self.input.cuda()
            # self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = len(self.train_X)
        self.perm = torch.randperm(self.ntrain)

    

    def fit(self):
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                batch_input = batch_input.cuda()
                batch_label = batch_label.cuda()
                batch_input.require_grad = True

                inputv = Variable(batch_input)
                labelv = Variable(batch_label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
                if i % 1000 == 0:
                    print('epoch finish:', epoch, 'iter finish:', i, "loss is:", loss)
            # print('epoch finish:', epoch)
            acc_per_class, whole_acc = self.cal_acc()
            print("In epoch:", epoch, "accuracy is:", acc_per_class, whole_acc)
        self.index_in_epoch = 0

    def cal_acc(self):
        predicted_label = []
        target_classes = list(range(0, self.nclass))
        ground_label = []
        for i in range(0, self.ntrain, self.batch_size):
            batch_input, batch_label = self.next_batch(self.batch_size)
            batch_input = batch_input.cuda()
            batch_label = batch_label.cpu()
            batch_input.require_grad = True

            inputv = Variable(batch_input)
            # labelv = Variable(batch_label)
            output = self.model(inputv)
            _, loc = torch.max(output.data, 1)
            predicted_label.append(loc.cpu())
            ground_label.append(batch_label)
        acc_per_class, whole_acc = self.compute_per_class_acc_gzsl(ground_label, predicted_label, target_classes)
        return acc_per_class, whole_acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        cor_per_class = torch.zeros(len(target_classes))
        sum_per_class = torch.zeros(len(target_classes))
        test_label = torch.tensor(test_label)
        predicted_label = torch.tensor(predicted_label)
        sample_size = predicted_label.size(0)
        for i in range(sample_size):
            sum_per_class[test_label[i]] += 1.0
            if test_label[i] == predicted_label[i]:
                cor_per_class[test_label[i]] += 1.0
        acc_per_class = torch.div(cor_per_class, sum_per_class)
        whole_acc = torch.div(cor_per_class.sum(), sum_per_class.sum())
        return acc_per_class, whole_acc


    def next_batch(self, batch_size):
        start = self.index_in_epoch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            # shuffle the data
            self.perm = torch.randperm(self.ntrain)
            print("---------alter once-----------------------------------------------------")
            # start next epoch
            self.index_in_epoch = 0
            res_X = self.train_X[self.perm[self.index_in_epoch]]
            res_Y = self.train_Y[self.perm[self.index_in_epoch]]
            self.index_in_epoch += batch_size
            return res_X, res_Y
        else:
            res_X = self.train_X[self.perm[self.index_in_epoch]]
            res_Y = self.train_Y[self.perm[self.index_in_epoch]]
            self.index_in_epoch += batch_size
            # from index start to index end-1
            return res_X, res_Y

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda(), volatile=True)) 
            else:
                output = self.model(Variable(test_X[start:end], volatile=True)) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        return acc_per_class.mean() 

# class FineTuned_ResNet_Classifer(nn.Module):
#     def __init__(self, num_classes):
#         super(FineTuned_ResNet_Classifer, self).__init__()
#         self.submodel = models.resnet18(pretrained=True)
#         feature_extract = True
#         self.submodel = self.set_parameter_requires_grad(self.submodel, feature_extract)
#         num_ftrs = self.submodel.fc.in_features
#         self.submodel.fc = nn.Linear(num_ftrs, num_classes)
#         self.input_size = 224
#         self.logic = nn.LogSoftmax(dim=1)
#         # self.upsample = nn.functional.upsample(size = self.input_size, mode = 'nearest')
#
#
#     def set_parameter_requires_grad(self, model, feature_extracting):
#         if feature_extracting:
#             for param in model.parameters():
#                 param.requires_grad = False
#         return model
#
#     def forward(self, x):
#         x = nn.functional.upsample(x, size = self.input_size, mode = 'bilinear')
#         # x = self.upsample(x)
#         x = self.submodel(x)
#         o = self.logic(x)
#         return o

class NLayerDiscriminator_shape(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, pool_dim = 8, num_class = 8, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator_shape, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers


        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        sequence += [[nn.AdaptiveAvgPool2d((pool_dim, pool_dim))]]

        self.fc = nn.Linear(pool_dim*pool_dim, num_class)
        self.logic = nn.LogSoftmax(dim=1)


        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        input = self.model(input).view(1, -1)
        input = self.fc(input)
        input = self.logic(input)
        return input


# class DiscriminatorGAP(nn.Module):
#     """Discriminator. PatchGAN."""
#     def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=3, init_stride=1,  max_filters=None, nc=3, use_bnorm=False):
#         super(DiscriminatorGAP, self).__init__()
#
#         layers = []
#         self.nc=nc
#         self.c_dim = c_dim
#         layers.append(nn.Conv2d(nc, conv_dim, kernel_size=3, stride=1, padding=1))
#         layers.append(nn.BatchNorm2d(conv_dim))
#         layers.append(nn.LeakyReLU(0.1, inplace=True))
#         layers.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1))
#         layers.append(nn.MaxPool2d(2))
#         if use_bnorm:
#             layers.append(nn.BatchNorm2d(conv_dim))
#         layers.append(nn.LeakyReLU(0.1, inplace=True))
#
#         curr_dim = conv_dim
#         for i in range(1, repeat_num):
#             out_dim =  curr_dim*2 if max_filters is None else min(curr_dim*2, max_filters)
#             layers.append(nn.Conv2d(curr_dim, out_dim, kernel_size=3, stride=1, padding=1))
#             layers.append(nn.BatchNorm2d(out_dim))
#             layers.append(nn.LeakyReLU(0.1, inplace=True))
#             layers.append(ResidualBlockBnorm(dim_in=out_dim, dilation=1, padtype='zero'))
#             if (i < 4):
#                 # We want to have 8x8 resolution vefore GAP input
#                 layers.append(nn.MaxPool2d(2))
#             curr_dim = out_dim
#
#         self.main = nn.Sequential(*layers)
#         self.globalPool = nn.AdaptiveAvgPool2d(1)
#         self.classifyFC = nn.Linear(curr_dim, c_dim, bias=False)
#
#     def forward(self, x, label=None):
#         bsz = x.size(0)
#         sz = x.size()
#         h = self.main(x)
#         out_aux = self.classifyFC(self.globalPool(h).view(bsz, -1))
#         return None, out_aux.view(bsz,self.c_dim)
#
#
# class ResidualBlockBnorm(nn.Module):
#     """Residual Block."""
#     def __init__(self, dim_in, dilation=1, padtype = 'zero'):
#         super(ResidualBlockBnorm, self).__init__()
#         pad = dilation
#         layers = []
#         if padtype == 'reflection':
#             layers.append(nn.ReflectionPad2d(pad)); pad=0
#         elif padtype == 'replication':
#             layers.append(nn.ReplicationPad2d(pad)); pad=0
#
#         layers.extend([ nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
#             nn.BatchNorm2d(dim_in, affine=True),
#             nn.LeakyReLU(0.1,inplace=True)])
#
#         pad = dilation
#         if padtype== 'reflection':
#             layers.append(nn.ReflectionPad2d(pad)); pad=0
#         elif padtype == 'replication':
#             layers.append(nn.ReplicationPad2d(p)); pad=0
#
#         layers.extend([
#             nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
#             nn.BatchNorm2d(dim_in, affine=True),
#             nn.LeakyReLU(0.1,inplace=True)
#             ])
#
#         self.main = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return x + self.main(x)