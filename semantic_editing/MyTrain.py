import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import scipy.io as io
import json
import MyFunc

# 上述描写中表明使用BCE_loss会导致不稳定的求导，这时使用BCEWithLogitsLoss()函数即可。
opt = TrainOptions().parse()  # 进行参数的设置和写入这些参数去opt.txt文件中
# opt.use_full_data = False
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)  # 打印频率，乘积和最大公约数的商
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

# -------------------------------
if opt.is_scGraph == 2:
    DesPath = './datasets/cityscapes/train_scGraph/'
    # DesPath = './datasets2/cityscapes/train_scGraph/'
    FileName = 'Shape2Set' + '2975' + '.json'
    DesFile = DesPath + FileName
    ReadFile_scGraph = open(DesFile, 'r', encoding='utf-8')
    Read_scGraph = json.load(ReadFile_scGraph)
    print("Test Read Scene Graph Successfully!")

DesPath = './datasets/cityscapes/train_scGraph/'
# FileName = 'ShortTransformMeanShapeSet'+str(dataset_size) +'.json'
FileName = 'ShortTransformMeanShapeSet' + '2975' + '.json'
DesFile = DesPath + FileName

ReadFile_MeanShape = open(DesFile, 'r', encoding='utf-8')
MeanShape_dict2 = json.load(ReadFile_MeanShape)
print("Test Read MeanShape_dict2 Successfully!")

SuitableObjeID = eval(opt.SuitableObjeID)

if opt.is_ClassiForShape == 1:
    if opt.dataName == 'cityscape':
        if opt.gpu_ids[0] == 1:
            pretrain_classifier_path2 = 'ED_pretrained_classifier_gpu1.pkl'
        elif opt.gpu_ids[0] == 0:
            pretrain_classifier_path2 = 'ED_pretrained_classifier_gpu0.pkl'
        pretrained_classifier = torch.load(pretrain_classifier_path2)
    elif opt.dataName == 'NYU':
        if opt.gpu_ids[0] == 1:
            pretrain_classifier_path2 = 'ED_pretrained_classifier_gpu1_NYU.pkl'
        elif opt.gpu_ids[0] == 0:
            pretrain_classifier_path2 = 'ED_pretrained_classifier_gpu0_NYU.pkl'
        pretrained_classifier = torch.load(pretrain_classifier_path2)
    elif opt.dataName == 'helen':
        if opt.gpu_ids[0] == 1:
            pretrain_classifier_path2 = 'ED_pretrained_classifier_gpu1_Helen.pkl'
        elif opt.gpu_ids[0] == 0:
            pretrain_classifier_path2 = 'ED_pretrained_classifier_gpu0_Helen.pkl'
        pretrained_classifier = torch.load(pretrain_classifier_path2)

    for p in pretrained_classifier.model.parameters():  # set requires_grad to False
        p.requires_grad = False

    projected_label = range(0, len(SuitableObjeID))
    SuitableObjeID2ProjectedLabel = {}
    ProjectedLabel2SuitableObjeID = {}

    for i in range(len(projected_label)):
        SuitableObjeID2ProjectedLabel[str(SuitableObjeID[i])] = projected_label[i]
        ProjectedLabel2SuitableObjeID[i] = str(SuitableObjeID[i])
else:
    pretrained_classifier = None
    SuitableObjeID2ProjectedLabel = None
    ProjectedLabel2SuitableObjeID = None

removalOjbID = 'random'
randomSeedID = None
PredefinedSize = -1

# Start import my own scGraph for all images one by one
if opt.use_full_data:
    DesPath = './datasets2/cityscapes/train_scGraph/'
else:
    DesPath = './datasets/cityscapes/train_scGraph/'

if opt.is_scGraph == 2:
    FileName = 'train_scGraph' + str(dataset_size) + '.json'
    DesFile = DesPath + FileName
    ReadFile_scGraph = open(DesFile, 'r', encoding='utf-8')
    Read_scGraph = json.load(ReadFile_scGraph)

    BinAdjFileName = 'train_scAdj' + 'BinaryAdjSet' + str(dataset_size) + '.npy'
    BinDesAdjSetFile = DesPath + BinAdjFileName
    RealAdjFileName = 'train_scAdj' + 'RealAdjSet' + str(dataset_size) + '.npy'
    RealDesAdjSetFile = DesPath + RealAdjFileName

    BinaryAdjSet = np.load(BinDesAdjSetFile)
    RealAdjSet = np.load(RealDesAdjSetFile)
    print("Test Read Scene Graph Successfully!")

# AdjFileName = 'train_scAdj' + str(dataset_size) + '.mat'
# DesAdjSetFile = DesPath + AdjFileName
# AdjSet = io.loadmat(DesAdjSetFile)

# classNum = 11   #对当前工作没用
# instNumEachClass = 25 #对当前工作没用


ExcVal = 35  # 35 for cityscape  895 for NYU 11 for Helen

# opt.numEdges = 3
# opt.numNodes = classNum * instNumEachClass + opt.numEdges


use_prior_obj_feature = False
# if not use_prior_obj_feature:
#     InitNodeEmb = torch.cuda.LongTensor(range(opt.numNodes))

# Finish import my own scGraph for all images one by one
# -------------------------------

model = create_model(opt)
# model.pretrained_cls = pretrained_classifier
# model.SeleL2ProjL = SuitableObjeID2ProjectedLabel
# model.ProjL2SeleL = ProjectedLabel2SuitableObjeID

visualizer = Visualizer(opt)

optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        if PredefinedSize == -1 and opt.objFilterCoe >= 0:
            InstSize = data['inst'].size()
            PredefinedSize = InstSize[2] * InstSize[3] * opt.objFilterCoe

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        if opt.is_scGraph == 2:  # 使用sceneGraph
            # ----------------------------------------------------------
            # Below is add by Jianfeng He
            CurPathKey = data['path'][0]
            TriScGraphList = Read_scGraph[CurPathKey]['scGraph']
            dataID = Read_scGraph[CurPathKey]['orderID']
            BinaryAdj = BinaryAdjSet[dataID, :, :]
            RealAdj = RealAdjSet[dataID, :, :]
            # BinaryAdj = AdjSet['BinaryAdjSet'][dataID, :, :]
            # RealAdj = AdjSet['RealAdjSet'][dataID, :, :]

            InComLableMap, InComInstMap, InComScGraph, InComBinAdj, InComRealAdj, InComImg, SelectedObj = \
                MyFunc.SelfSupervisePrePro_Advanced(data['label'], data['inst'], data['image'], data['feat'],
                                                    TriScGraphList,
                                                    classNum, instNumEachClass, ExcVal, removalOjbID='random',
                                                    randomSeedID=679)
            ComRealAdj = torch.from_numpy(RealAdj)

            # Above is added by Jianfeng He
            # ----------------------------------------------------------
            losses, generated = model(Variable(InComLableMap), Variable(InComInstMap), Variable(InComBinAdj),
                                      Variable(InComRealAdj), Variable(ComRealAdj), classNum, instNumEachClass,
                                      Variable(InComImg), Variable(data['image']), Variable(data['feat']),
                                      Variable(InitNodeEmb), infer=save_fake)
        elif opt.is_scGraph == 0:  # 使用原始的pix2pixHD,没有修改
            losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                      Variable(data['image']), Variable(data['feat']), infer=save_fake)
        elif opt.is_scGraph == 1:  # 使用手动输入一个item
            # ----------------------------------------------------------
            # Below is add by Jianfeng He
            try:
                InComLableMap, InComInstMap, InComImg, SelectedObj, SelectedCalss, SelectedBox, isNone = \
                    MyFunc.SelfSupervisePrePro_Basic(data['label'], data['inst'], data['image'], data['feat'],
                                                     SuitableObjeID, ExcVal, PredefinedSize, removalOjbID, randomSeedID)
            except:
                continue
            if isNone == True:
                continue

            if opt.is_shapePrior == 1:
                MapSize = InComLableMap.shape
                SelectedShape = MeanShape_dict2[SelectedCalss]
                SelectedObjLayer = MyFunc.PickupShape(SelectedShape, SelectedBox, MapSize)  # 物体形状的先验,但是此时此刻是01概率表示
            elif opt.is_shapePrior == 0:
                MapSize = InComLableMap.shape
                SelectedObjLayer = MyFunc.FillShapeAll0(SelectedBox, MapSize)

            # infer = True
            # Above is added by Jianfeng He
            # ----------------------------------------------------------

            losses, generated = model(Variable(InComLableMap), Variable(InComInstMap), Variable(SelectedObjLayer),
                                      Variable(InComImg), Variable(data['image']), Variable(data['feat']))

        elif opt.is_scGraph == 3:  # 使用手动输入一个item
            # ----------------------------------------------------------
            # Below is add by Jianfeng He
            # InComLableMap, InComInstMap, InComImg, SelectedObj, SelectedCalss, SelectedBox, isNone = \
            #     MyFunc.SelfSupervisePrePro_Basic(data['label'], data['inst'], data['image'], data['feat'],
            #                                      SuitableObjeID, ExcVal, PredefinedSize, removalOjbID, randomSeedID)
            try:
                # util.tensor2label(data['label'][0], opt.label_nc)
                InComLableMap, InComInstMap, InComImg, SelectedObj, SelectedCalss, SelectedBox, isNone, SpeciObjMap = \
                    MyFunc.SelfSupervisePrePro_Basic(data['label'], data['inst'], data['image'], data['feat'], SuitableObjeID, ExcVal, PredefinedSize, removalOjbID, randomSeedID)
            except:
                if i % 100 == 0:
                    print(i)
                    print("there is no suitalbe size object in the image, skip!!!", i)
                continue
            if isNone == True:
                print("SKIP!")
                continue


            if opt.is_shapePrior == 1:
                MapSize = InComLableMap.shape
                SelectedShape = MeanShape_dict2[SelectedCalss]  # 需要重新计算形状先验的数值111111
                SelectedObjLayer = MyFunc.PickupShape2(SelectedShape, SelectedBox, MapSize)  # 物体形状的先验,但是此时此刻是01概率表示
            elif opt.is_shapePrior == 0:
                MapSize = InComLableMap.shape
                FillVal = 1
                SelectedObjLayer = MyFunc.FillShapeAllSameVal(SelectedBox, MapSize, FillVal)
            # 开始拼接 IncomLabel IncomInst
            InComLableMap[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1] = int(SelectedCalss)
            InComInstMap[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1] = int(
                SelectedCalss) * 1000 + 100
            SelectedObjLayer = MyFunc.FillShapeAllSameVal(SelectedBox, MapSize, 1)

            # infer = True
            # Above is added by Jianfeng He
            # ---------------------------------------------------------- pretrained_classifier
            losses, generatedSet = model(Variable(InComLableMap), Variable(InComInstMap), Variable(SelectedObjLayer),
                                         Variable(InComImg), Variable(data['image']), SelectedBox,
                                         pretrained_classifier, SuitableObjeID2ProjectedLabel, int(SelectedCalss),
                                         Variable(data['feat']))
            # try:
            #     losses, generatedSet = model(Variable(InComLableMap), Variable(InComInstMap), Variable(SelectedObjLayer), Variable(InComImg), Variable(data['image']), SelectedBox, pretrained_classifier, SuitableObjeID2ProjectedLabel, int(SelectedCalss), Variable(data['feat']))
            # except:
            #     print('something wrong!')
            #     print('image index:', i)
            #     print('selected obj:', SelectedObj)
            #     continue

            generated = generatedSet[0]
            ori_pls_gen = generatedSet[1]

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar   'Loc_G_Shp'
        if opt.MultiExpanTimes <= 0 and opt.is_ClassiForShape == 0:
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)
        elif opt.MultiExpanTimes > 0 or opt.is_ClassiForShape == 1:
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real'] + opt.labmdaMulExp * loss_dict.get('Loc_D_fa',
                                                                                                   0) + opt.labmdaMulExp * loss_dict.get(
                'Loc_D_re', 0)) * 0.25
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG',
                                                                                         0) + opt.labmdaMulExp * loss_dict.get(
                'Loc_G_gan', 0) + opt.labmdaShape * loss_dict.get('Loc_G_Shp', 0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            # loss_G.backward(retain_graph=True)
            loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_D.backward()
        optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ### display output images     #根据该作者的return，此功能应该无法开启
        if save_fake:
            visuals = OrderedDict([('Complete_label_map', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('Incomplete_label_map', util.tensor2label(InComLableMap[0], opt.label_nc)),
                                   ('Synthesized_image', util.tensor2im(generated.data[0])),
                                   ('MaskPlusSyn_image', util.tensor2im(ori_pls_gen.data[0])),
                                   ('Real_image', util.tensor2im(data['image'][0])),
                                   ('InCom_img', util.tensor2im(InComImg[0])),
                                   ])
            visualizer.display_current_results(visuals, epoch, total_steps, SelectedObj)
            # print('showing image removes:', SelectedObj)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
