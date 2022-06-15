import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')   #256 Cityscape #2048 #192 NYU
        self.parser.add_argument('--fineSize', type=int, default=128, help='then crop to this size')      #128 Cityscape #1024 #96 NYU
        self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets2/cityscapes/')   #此选项可以用来开启是toy还是full data
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        #for scGraph usage
        self.parser.add_argument('--is_scGraph', type = int, default=1,
                                 help='0: original pix2pixHD; 1: reviseApp is one single item; 2: reviseApp is scGraph; 3: ECCV model')
        self.parser.add_argument('--NodeInputDim', type=int, default=128, help='initialized node embedding dimensions, F0')
        self.parser.add_argument('--NodeHiddenDim', type=int, default=256, help='hidden state for node embedding dimensions, F1')
        self.parser.add_argument('--NodeOutputDim', type=int, default=256, help='output dimension of EGCN for node embedding dimensions, F2')
        self.parser.add_argument('--EdgeEmbDim', type=int, default=128, help='output dimension of EGCN for Edge embedding dimensions, P')
        # self.parser.add_argument('--use_full_data', type=bool, default=False,
        #                          help='if set as True, then the model will use all data rather than patial data')
        self.parser.add_argument('--use_full_data', action='store_true',
                                 help='if set as True, then the model will use all data rather than patial data')

        # '[24, 25, 26, 27, 28, 31, 32, 33]'
        self.parser.add_argument('--SuitableObjeID', type = str, default= '[24, 25, 26, 27, 28, 31, 32, 33]',
                                 help='if set as True, then the model will use all data rather than patial data')
        self.parser.add_argument("--Te", action='store_true', help='if specified, it is Test mode')
        self.parser.add_argument('--is_shapePrior', type=int, default=1,
                                 help='0: blank; 1: possible; 2: apply neighbor information')
        #0.02
        self.parser.add_argument('--objFilterCoe', type=float, default=0.02,
                                 help='0: can choose any size shape; >0: only choose size bigger than this value;')
        self.parser.add_argument('--MultiExpanTimes', type=int, default=0,
                                 help='0: No Multi Expansion will be done; >0: Do the multi expansion so many times;')
        self.parser.add_argument('--MultiExpanRadius', type=int, default=0,
                                 help='0: Multi Expansion Radius for every time; >0: only choose size bigger than this value;')
        self.parser.add_argument('--ImageFileEnd', type=str, default='_img',
                                 help='imgFile end forms, they can be _img, _img2labelcolor, _img2labelgray')
        self.parser.add_argument('--is_ClassiForShape', type=int, default=0,
                                 help='0: Do NOT use it; 1: Apply this kind of loss;')
        self.parser.add_argument('--is_Toy', type=int, default=0,
                                 help='0: use full data; 1: Apply toy data;')
        self.parser.add_argument('--dataName', type=str, default='cityscape')  # 此选项可以用来开启是toy还是full data
        self.parser.add_argument('--labmdaShape', type=float, default=1.0, help='setting for the coeffiecnet of shape loss')
        self.parser.add_argument('--labmdaMulExp', type=float, default=1.0,
                                 help='setting for the coeffiecnet of multiExpansion loss')
        #aaai prepared
        self.parser.add_argument('--global_mask', type=int, default=1,
                                 help='setting applying gloable mask for 1 or local(concorete bounding box) mask for 0.')
        self.parser.add_argument('--nip18model', type=int, default=1,
                                 help='setting applying gloable mask for applying nips18 model or eccv20 model for 0.')
        self.parser.add_argument('--MEx_approx', action='store_true',
                                 help='apply MEx approximation by only 1 discriminator, if false, apply original MEx with multi Discriminator')

        self.parser.add_argument('--not_sum_by_mask', action='store_true',
                                 help='do not sum by mask if apply the option')

        # self.initialized = True
        self.initialized = True


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
