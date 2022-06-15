import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TrainOptions().parse()     #进行参数的设置和写入这些参数去opt.txt文件中
# opt.serial_batches = True  # no shuffle
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    #打印频率，乘积和最大公约数的商
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# model = create_model(opt)
visualizer = Visualizer(opt)
# if opt.fp16:
#     from apex import amp
#     model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')
#     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
# else:
#     optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

key = "train_"
desFolder = "./datasets2/HelenFace/"
import scipy.misc


for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):

        labelMap = util.tensor2label(data['label'][0], opt.label_nc)
        imageMap = util.tensor2im(data['image'][0])
        labelFileName = desFolder + key + 'maskedlabel/' + 'Helen' + str(i) + '_gtFine_labelIds.png'
        imageFileName = desFolder + key + 'maskedimg/' + 'Helen' + str(i) + '_leftImg8bit.png'
        scipy.misc.imsave(labelFileName, labelMap)
        scipy.misc.imsave(imageFileName, imageMap)
        print(i)
