import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import json
import MyFunc
import copy
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from sklearn import metrics

from PIL import Image


def save_img(image_tensor, filename):
    image_numpy = image_tensor
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#Testing Images = %d' % dataset_size)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# below is added by Jianfeng
DesPath = './datasets/cityscapes/train_scGraph/'
# FileName = 'ShortTransformMeanShapeSet'+str(dataset_size) +'.json'
FileName = 'ShortTransformMeanShapeSet' + '2975' + '.json'
DesFile = DesPath + FileName

ReadFile_MeanShape = open(DesFile, 'r', encoding='utf-8')
MeanShape_dict2 = json.load(ReadFile_MeanShape)
print("Test Read MeanShape_dict2 Successfully!")

SuitableObjeID = eval(opt.SuitableObjeID)
ExcVal = 35  # city might be 35/36, helen 11 NYU895
class_N = 36  # NYU 895/896 cityscapes_newcolor 36   Helen 12 cityscapes_ori_color 35
#

removalOjbID = 'random'
randomSeedID = 679
PredefinedSize = -1

MSE_list = []
MSE_list_label = []
mIOU_ChoseObj_list = []
mIOU_AllObj_list = []

if opt.is_ClassiForShape == 1:
    shape_GT = []
    shape_Te = []
    if opt.dataName == 'cityscape':
        if opt.gpu_ids[0] == 1:
            pretrain_classifier_path2 = 'TE_pretrained_classifier_gpu1.pkl'
        elif opt.gpu_ids[0] == 0:
            pretrain_classifier_path2 = 'TE_pretrained_classifier_gpu0.pkl'
        pretrained_classifier = torch.load(pretrain_classifier_path2)
    elif opt.dataName == 'NYU':
        if opt.gpu_ids[0] == 1:
            pretrain_classifier_path2 = 'TE_pretrained_classifier_gpu1_NYU.pkl'
        elif opt.gpu_ids[0] == 0:
            pretrain_classifier_path2 = 'TE_pretrained_classifier_gpu0_NYU.pkl'
        pretrained_classifier = torch.load(pretrain_classifier_path2)

    elif opt.dataName == 'helen':
        if opt.gpu_ids[0] == 1:
            pretrain_classifier_path2 = 'TE_pretrained_classifier_gpu1_Helen.pkl'
        elif opt.gpu_ids[0] == 0:
            pretrain_classifier_path2 = 'TE_pretrained_classifier_gpu0_Helen.pkl'
        pretrained_classifier = torch.load(pretrain_classifier_path2)

    for p in pretrained_classifier.model.parameters():  # set requires_grad to False
        p.requires_grad = False

    projected_label = range(0, len(SuitableObjeID))
    SuitableObjeID2ProjectedLabel = {}
    ProjectedLabel2SuitableObjeID = {}

    for i in range(len(projected_label)):
        SuitableObjeID2ProjectedLabel[str(SuitableObjeID[i])] = projected_label[i]
        ProjectedLabel2SuitableObjeID[i] = str(SuitableObjeID[i])
    print('load shape pretrained classifier finished!')
else:
    pretrained_classifier = None
    SuitableObjeID2ProjectedLabel = None
    ProjectedLabel2SuitableObjeID = None

# above is added by Jianfeng


# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)

    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

save_res = 0

skip_jsq = 0

if save_res:

    effec_jsq = 0
    # first_order_path = '/home/jfhe/Desktop/ECCV_AMT/Pix2PixHD_MEx/Sum_aaai'

    MethodName = opt.name  # 'MExGAN' 'Pix2PixHD'
    first_order_path = '/home/jfhe/Desktop/ECCV_AMT/' + MethodName
    if not os.path.exists(first_order_path):
        os.mkdir(first_order_path)
    if not os.path.exists(first_order_path + '/generate'):
        os.mkdir(first_order_path + '/generate')
    if not os.path.exists(first_order_path + '/groundtruth'):
        os.mkdir(first_order_path + '/groundtruth')

for i, data in enumerate(dataset):
    print(i)
    if PredefinedSize == -1 and opt.objFilterCoe >= 0:
        InstSize = data['inst'].size()
        PredefinedSize = InstSize[2] * InstSize[3] * opt.objFilterCoe

    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst'] = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst'] = data['inst'].uint8()
    if opt.export_onnx:
        print("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    elif opt.is_scGraph == 1:  # 使用手动输入一个item
        try:
            InComLableMap, InComInstMap, InComImg, SelectedObj, SelectedCalss, SelectedBox, isNone = \
                MyFunc.SelfSupervisePrePro_Basic(data['label'], data['inst'], data['image'], data['feat'],
                                                 SuitableObjeID, ExcVal, removalOjbID='random', randomSeedID=None)
        except:
            skip_jsq = skip_jsq + 1
            print('skip !!!!')
            continue

        if isNone == True:
            print('SKIP ~~~~')
            continue

        MapSize = InComLableMap.shape
        SelectedShape = MeanShape_dict2[SelectedCalss]
        SelectedObjLayer = MyFunc.PickupShape(SelectedShape, SelectedBox, MapSize)  # 物体形状的先验

        generated = model.inference(Variable(InComLableMap), Variable(InComInstMap), Variable(SelectedObjLayer),
                                    Variable(InComImg), Variable(data['image']), Variable(data['feat']))

    elif opt.is_scGraph == 3:  # 使用手动输入一个item
        try:
            InComLableMap, InComInstMap, InComImg, SelectedObj, SelectedCalss, SelectedBox, isNone, SpeciObjMap = \
                MyFunc.SelfSupervisePrePro_Basic(data['label'], data['inst'], data['image'], data['feat'],
                                                 SuitableObjeID, ExcVal, PredefinedSize, removalOjbID, randomSeedID)

        except:
            skip_jsq = skip_jsq + 1
            print('skip !!!!')
            try:
                print(data['path'])
            except:
                print("No path for current dataset ")
            continue

        # InComLableMap, InComInstMap, InComImg, SelectedObj, SelectedCalss, SelectedBox, isNone = MyFunc.SelfSupervisePrePro_Basic(data['label'], data['inst'], data['image'], data['feat'], SuitableObjeID, ExcVal, PredefinedSize, removalOjbID, randomSeedID)

        if isNone == True:
            print('SKIP')
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
        # if SelectedCalss == '24':
        #     SelectedCalss = '26'
        # # elif SelectedCalss == '26':
        # #     SelectedCalss = '24'
        InComLableMap[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1] = int(SelectedCalss)
        InComInstMap[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1] = int(
            SelectedCalss) * 1000 + 100
        SelectedObjLayer = MyFunc.FillShapeAllSameVal(SelectedBox, MapSize, 1)

        res = model.inference(Variable(InComLableMap), Variable(InComInstMap), Variable(SelectedObjLayer),
                              Variable(InComImg), Variable(data['image']), SelectedBox, Variable(data['feat']))
        generated = res[0]
        mask_plus_generated = res[1]

        # 以下为校准数据到标准的ｃｏｌｏｒｍａｐ
        l_img_index = util.labelcolormap(class_N)
        # l_img_index = util.labelcolormap(36)

        # start generate labelMap
        Gen_ComLabelMap = torch.zeros_like(InComLableMap)
        chosen_gen = mask_plus_generated
        Ini_G_ComLabelMap = util.tensor2im(chosen_gen.data[0])
        wid, hig, cha = Ini_G_ComLabelMap.shape
        l_img_index = np.array(l_img_index).astype(np.float32)
        for ii in range(wid):
            for jj in range(hig):
                mid = np.array([Ini_G_ComLabelMap[ii, jj, :]]).astype(np.float32)
                mid2 = np.tile(mid, (class_N, 1))
                row_sub = l_img_index - mid2
                # row_pow = row_sub * row_sub
                row_pow = np.multiply(row_sub, row_sub)
                row_sum = row_pow.sum(axis=1)
                mid_loc = np.argmin(row_sum)
                Gen_ComLabelMap[:, :, ii, jj] = mid_loc.astype(float)
        Gen_F_ComLMap = copy.deepcopy(InComLableMap)
        Gen_F_ComLMap[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1] = Gen_ComLabelMap[:, :,
                                                                                            SelectedBox[0]:SelectedBox[
                                                                                                1]+1,
                                                                                            SelectedBox[2]:SelectedBox[
                                                                                                3]+1]
        # Gen_F_LocalMap = torch.zeros(1, 1, SelectedBox[1] - SelectedBox[0], SelectedBox[3] - SelectedBox[2])
        Gen_F_LocalMap = Gen_ComLabelMap[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1]
        Ori_R_LocalMap = data['label'][:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]+1:SelectedBox[3]+1]

        # Gen_F_ComIMap = copy.deepcopy(InComInstMap)
        # Gen_F_ComIMap[:, :, SelectedBox[0]:SelectedBox[1], SelectedBox[2]:SelectedBox[3]] = Gen_ComLabelMap[:, :, SelectedBox[0]:SelectedBox[1], SelectedBox[2]:SelectedBox[3]]

        # 以上为校准数据到标准的ｃｏｌｏｒｍａｐ

        local_part_image = util.tensor2label(Gen_F_LocalMap[0], opt.label_nc)  # 0-255 三通道 36 for my setting, original is 35 for second parameter
        # 以下为计算shape classification
        # local_part_image2 = F.to_pil_image(local_part_image.transpose(2, 0, 1))
        # local_part_image2 = F.to_pil_image(local_part_image.transpose(2, 0, 1))
        B = Image.fromarray(local_part_image.astype('uint8'))
        B = B.convert('RGB')
        transform_list = []
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        local_part_image_nor = transforms.Compose(transform_list)(B)  # 预期应该是 -1 ~ 1之间的一个三通道
        local_part_image_nor = torch.tensor(local_part_image_nor).unsqueeze(0)
        local_part_image_shape = MyFunc.pt_get_edges(local_part_image_nor)
        # mid_shape_res = pretrained_classifier.model(local_part_image_shape.cuda())
        # shape_val, shape_loc = torch.max(mid_shape_res, 1)
        # shape_Te.append(shape_loc.tolist()[0])
        # shape_GT.append(SuitableObjeID2ProjectedLabel[SelectedCalss])
        # 以上为计算shape classification

        # 以下为计算色彩MSE
        local_part_image_real = data['image'][:, :, SelectedBox[0]:SelectedBox[1]+1,
                                SelectedBox[2]:SelectedBox[3]+1]  # 应该是-1~1
        mid_MSE = 0
        for cha in range(local_part_image_real.shape[1]):
            mid_loc_real = local_part_image_real[:, cha, :, :].squeeze(0)
            mid_loc_gen = local_part_image_nor[:, cha, :, :].squeeze(0)
            mid_MSE += metrics.mean_squared_error(mid_loc_real, mid_loc_gen)
        MSE_list.append(mid_MSE)
        # 以上为计算色彩MSE

        # 以下为计算label_Hamming
        local_part_image_real = data['label'][:, :, SelectedBox[0]:SelectedBox[1]+1,
                                SelectedBox[2]:SelectedBox[3]+1]  # 应该是-1~1
        mid_MSE2 = 0
        local_area = local_part_image_real.shape[2] * local_part_image_real.shape[3]
        for cha in range(local_part_image_real.shape[1]):
            mid_loc_real = local_part_image_real[:, cha, :, :].squeeze(0)
            mid_loc_gen = Gen_F_LocalMap[:, cha, :, :].squeeze(0)
            mid_MSE2 += ((mid_loc_real == mid_loc_gen).sum().float() / local_area).tolist()
        MSE_list_label.append(mid_MSE2)
        # 以上为计算label_Hamming

        # 以下为计算mIOU
        local_part_label_real = data['label'][:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1]
        local_part_label_fake = Gen_F_LocalMap
        iouWiNan, iouWoNan = MyFunc.iou(local_part_label_fake[0, 0, :, :].contiguous(),
                                        local_part_label_real[0, 0, :, :].contiguous(), class_N, False)

        mIOU_ChoseObj_list.append(iouWiNan[int(SelectedCalss)])
        mIOU_AllObj_list.append(iouWoNan.sum() / len(iouWoNan))
        # 以上为计算mIOU
    # opt.label_nc = 35
    # 存储图片
    visuals = OrderedDict([('SynMask_image_colormap', util.tensor2label(Gen_F_ComLMap[0], opt.label_nc)),
                           ('ComSeg_colormap', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('Incomplete_label_map', util.tensor2label(InComLableMap[0], opt.label_nc)),
                           ('SyntMask_image_NoColormap', util.tensor2im(mask_plus_generated.data[0])),
                           ('SynLocal_colormap', util.tensor2label(Gen_F_LocalMap[0], opt.label_nc)),
                           ('OriLocal_Map', util.tensor2label(Ori_R_LocalMap[0], opt.label_nc)),
                           ('Real_image', util.tensor2im(data['image'][0])),
                           ('InCom_img', util.tensor2im(InComImg[0])),
                           ])
    if save_res:
        # second_order_path =  first_order_path + '/' + str(effec_jsq)
        # if not os.path.exists(second_order_path):
        #     os.mkdir(second_order_path)
        #
        # save_img(util.tensor2label(data['label'][0], opt.label_nc), (second_order_path + "/{}_{}_{}.jpg").format(str(effec_jsq), 'GroundTruth', SelectedCalss))
        # save_img(util.tensor2label(Gen_F_ComLMap[0], opt.label_nc),
        #          (second_order_path + "/{}_{}_{}.jpg").format(str(effec_jsq), MethodName, SelectedCalss))
        # save_img(util.tensor2label(InComLableMap[0], opt.label_nc),
        #          (second_order_path + "/{}_{}_{}.jpg").format(str(effec_jsq), 'Incomplete', SelectedCalss))

        save_img(util.tensor2label(data['label'][0], opt.label_nc),
                 (first_order_path + '/groundtruth' + "/{}_{}.jpg").format(str(i), "gt"))
        save_img(util.tensor2label(Gen_F_ComLMap[0], opt.label_nc),
                 (first_order_path + '/generate' + "/{}_{}.jpg").format(str(i), "gen"))

        effec_jsq = effec_jsq + 1

    # img_path = data['path']
    # print('process image... %s' % img_path)
    visualizer.display_current_results(visuals, i, 1, SelectedObj)
    print('showing image removes:', SelectedObj)

# 计算shape accuracy
# shape_accuracy = metrics.accuracy_score(shape_GT, shape_Te, normalize=False)
# shape_macro_pre = metrics.precision_score(shape_GT, shape_Te, average='macro')
# shape_micro_pre = metrics.precision_score(shape_GT, shape_Te, average='micro')
# shape_macro_rec = metrics.recall_score(shape_GT, shape_Te, average='macro')
# shape_micro_rec = metrics.recall_score(shape_GT, shape_Te, average='micro')
# shape_F1 = metrics.f1_score(shape_GT, shape_Te, average='weighted')

# print("model_name", opt.name)
# print("shape_accuracy", shape_accuracy)
# print("shape_macro_pre", shape_macro_pre)
# print("shape_micro_pre", shape_micro_pre)
# print("shape_macro_rec", shape_macro_rec)
# print("shape_micro_rec", shape_micro_rec)
# print("shape_F1", shape_F1)

# 计算AverageMSE
print("AverageMSe_ImageLevel", np.sum(np.array(MSE_list)) / len(MSE_list))
print("AverageHammingLoss_LabelLevel", np.sum(np.array(MSE_list_label)) / len(MSE_list_label))

# 计算mIOUbuntu
print("Choose_Obj_mIOU", np.sum(np.array(mIOU_ChoseObj_list)) / len(mIOU_ChoseObj_list))
print("All_Obj_mIOU", np.sum(np.array(mIOU_AllObj_list)) / len(mIOU_AllObj_list))

# Shape accuracy 和 AverageMSE是从三通道 -1 ~ 1 的局部来判断的
# mIOU是从 单通道 0~N_class来判断

print('total skip number is:', skip_jsq)

WriteFileName = opt.checkpoints_dir + '/' + opt.name + '/' + 'MSEres.txt'
with open(WriteFileName, 'w') as f:
    # f.write("shape_accuracy:" + str(shape_accuracy) + '\n')
    # f.write("shape_macro_pre:" + str(shape_macro_pre) + '\n')
    # f.write("shape_micro_pre:" + str(shape_micro_pre) + '\n')
    # f.write("shape_macro_rec:" + str(shape_macro_rec) + '\n')
    # f.write("shape_micro_rec:" + str(shape_micro_rec) + '\n')
    # f.write("shape_F1:" + str(shape_F1) + '\n')
    f.write("AverageMSe_ImageLevel" + str(np.sum(np.array(MSE_list)) / len(MSE_list)) + '\n')
    f.write("AverageHammingLoss_LabelLevel" + str(np.sum(np.array(MSE_list_label)) / len(MSE_list_label)) + '\n')
    f.write("Choose_Obj_mIOU" + str(np.sum(np.array(mIOU_ChoseObj_list)) / len(mIOU_ChoseObj_list)) + '\n')
    f.write("All_Obj_mIOU" + str(np.sum(np.array(mIOU_AllObj_list)) / len(mIOU_AllObj_list)) + '\n')
    f.write("total skip number is:" + str(skip_jsq) + '\n')
f.close()

webpage.save()

