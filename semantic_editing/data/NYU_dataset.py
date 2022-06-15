import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import mat73
import numpy as np
import torch
from PIL import Image
import scipy
import util.util as util
import copy


class NYUDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input B (real images)
        if opt.is_Toy == 1:
            fileName = 'nyu_depth_v2_toy_labeled.mat'
            self.dir_B = os.path.join(opt.dataroot, fileName)
            self.mat = scipy.io.loadmat(self.dir_B)

            segFName = 'nyu_depth_v2_labeled_processed_toy.npy'
            self.dir_seg = os.path.join(opt.dataroot, segFName)
            self.segMat = np.load(self.dir_seg)
        else:
            fileName = 'nyu_depth_v2_labeled.mat'
            self.dir_B = os.path.join(opt.dataroot, fileName)
            self.mat = mat73.loadmat(self.dir_B)
            self.mat['labels'] = self.mat['labels'].transpose()
            self.mat['instances'] = self.mat['instances'].transpose()
            self.mat['images'] = self.mat['images'].transpose()

            segFName = 'nyu_depth_v2_labeled_processed.npy'
            self.dir_seg = os.path.join(opt.dataroot, segFName)
            self.segMat = np.load(self.dir_seg)






        H, W, self.TotalSampleNum = self.mat['labels'].shape

        if opt.is_Toy == 1:
            self.TrSampleNum = 6
            self.TeSampleNum = 2
        else:
            self.TrSampleNum = 1200
            self.TeSampleNum = 249


      
    def __getitem__(self, index):        
        ### input A (label maps)  B(imagas)
        if self.opt.isTrain:
            A = self.mat['labels'][:, :, range(self.TrSampleNum)[index]]
            B = self.segMat[:, :, :, range(self.TrSampleNum)[index]]
            inst = (self.mat['instances'][:, :, range(self.TrSampleNum)[index]] + A * 1000)
        elif self.opt.Te:
            A = self.mat['labels'][:, :, range(self.TrSampleNum, self.TrSampleNum + self.TeSampleNum)[index]]
            B = self.segMat[:, :, :, range(self.TrSampleNum, self.TrSampleNum + self.TeSampleNum)[index]]
            inst = (self.mat['instances'][:, :, range(self.TrSampleNum, self.TrSampleNum + self.TeSampleNum)[index]] + A * 1000)

        A = Image.fromarray(A.astype('uint8'))
        B = Image.fromarray(B.astype('uint8'))
        inst = Image.fromarray(inst.astype('uint32'))


        params = get_params(self.opt, A.size)    #得转换成 形状为w h的形式
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)


        B = B.convert('RGB')
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)

        ### if using instance maps
        if not self.opt.no_instance:
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor}

        return input_dict

    def __len__(self):
        if self.opt.isTrain:
            return self.TrSampleNum
        elif self.opt.Te:
            return  self.TeSampleNum

    def name(self):
        return 'NYU'