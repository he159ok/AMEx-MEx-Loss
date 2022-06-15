import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import copy
from .mask_losses import MaskReconLoss

class MyBasePix2PixHDModel(BaseModel):
	def name(self):
		return 'Pix2PixHDModel'

	def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
		flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)

		def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
			return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]

		return loss_filter


	def init_loss_filter_multi_exp(self, use_gan_feat_loss, use_vgg_loss):
		if self.opt.is_ClassiForShape == 0 and self.opt.MultiExpanTimes > 0:
			flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True, False)
		elif self.opt.is_ClassiForShape == 1 and self.opt.MultiExpanTimes > 0:
			flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True, True)
		elif self.opt.is_ClassiForShape == 1 and self.opt.MultiExpanTimes == 0:
			flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, False, False, False, True)
		else:
			assert 1 == 0
		def loss_filter_multi_expan(g_gan, g_gan_feat, g_vgg, d_real, d_fake, l_d_fa, l_d_re, l_g_gan, l_g_shp):
			return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake, l_d_fa, l_d_re, l_g_gan, l_g_shp), flags) if f]

		return loss_filter_multi_expan

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
			torch.backends.cudnn.benchmark = True
		self.opt = opt
		self.isTrain = opt.isTrain
		self.use_features = opt.instance_feat or opt.label_feat
		self.gen_features = self.use_features and not self.opt.load_features
		self.use_scGraph = opt.is_scGraph
		input_nc = (opt.label_nc + 1) if opt.label_nc != 0 else opt.input_nc
		# input_nc = (input_nc + 1) if opt.is_shapePrior == 1 else input_nc
		self.input_nc = input_nc
		# self.pretrained_cls = None
		# self.SeleL2ProjL = None
		# self.ProjL2SeleL = None
		# below is applied after aaai.
		self.criterionCtxRecon = MaskReconLoss()
		self.criterionObjRecon = torch.nn.BCELoss()
		# self.criterionObjRecon = torch.nn.BCEWithLogitsLoss()
		self.L1Loss_Global = torch.nn.L1Loss()
		self.L1Loss_Lobal = torch.nn.L1Loss()

		##### define networks
		# scGraph Related Network
		if self.is_scGraph == 2:
			self.NodeInputDim = opt.NodeInputDim
			self.NodeHiddenDim = opt.NodeHiddenDim
			self.NodeOutputDim = opt.NodeOutputDim
			self.EdgeEmbDim = opt.EdgeEmbDim
			self.numOfNodes = opt.numNodes
			self.numberOfEdges = opt.numEdges
			self.scEGCN = networks.define_EGCN(self.NodeInputDim, self.NodeHiddenDim, self.NodeOutputDim, self.EdgeEmbDim)
			self.scNodeEmb = networks.define_NodeEmb(self.numOfNodes, self.NodeInputDim)
			self.scEdgeEmb = networks.define_EdgeEmb(self.numberOfEdges + 1, self.EdgeEmbDim)

		# Generator network
		netG_input_nc = input_nc
		if not opt.no_instance:
			netG_input_nc += 1
		if self.use_features:
			netG_input_nc += opt.feat_num

		self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
		                              opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
		                              opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, my_opt=self.opt)

		# Discriminator network
		if self.isTrain:
			use_sigmoid = opt.no_lsgan
			netD_input_nc = input_nc + opt.output_nc
			if not opt.no_instance:
				netD_input_nc += 1
			self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
			                              opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

			if self.opt.MultiExpanTimes > 0:
				self.MultiExpan_netD = networks.define_D_MultiExpan(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
				                              opt.MultiExpanTimes, opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

			if self.opt.is_ClassiForShape== 1:
				self.cls_criterion = torch.nn.NLLLoss()

			#added by Jianfeng
			#---------------------------
			#增加一个新的判别模型

			#---------------------------
			#added by Jianfeng

		### Encoder network
		if self.gen_features:
			self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
			                              opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
		if self.opt.verbose:
			print('---------- Networks initialized -------------')

		# load networks
		if not self.isTrain or opt.continue_train or opt.load_pretrain:
			pretrained_path = '' if not self.isTrain else opt.load_pretrain
			self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
			if self.isTrain:
				self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
			if self.gen_features:
				self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

			# set loss functions and optimizers
		if self.isTrain:
			if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
				raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
			self.fake_pool = ImagePool(opt.pool_size)
			self.old_lr = opt.lr

			# define loss functions
			if opt.MultiExpanTimes <= 0 and self.opt.is_ClassiForShape == 0:
				self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
			elif opt.MultiExpanTimes > 0 or self.opt.is_ClassiForShape == 1:
				self.loss_filter_multi_expan = self.init_loss_filter_multi_exp(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

			self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
			# if opt.no_lsgan:
			# 	#如果不使用LSGAN,则设置 真的判别矩阵为全1,假的判别矩阵为全0  b=c=1 a=0 #符合LSGAN的setting
			# 	self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
			# else:
			# 	# 如果使用LSGAN,则设置 真的判别矩阵为全1,假的判别矩阵为全 -1  b=c=1 a = -1
			# 	self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, target_real_label=1.0, target_fake_label=-1.0, tensor=self.Tensor)

			self.criterionFeat = torch.nn.L1Loss()
			if not opt.no_vgg_loss:
				self.criterionVGG = networks.VGGLoss(self.gpu_ids)

			# Names so we can breakout loss
			if self.opt.MultiExpanTimes <= 0 and self.opt.is_ClassiForShape == 0:
				self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')
			elif self.opt.MultiExpanTimes > 0 or self.opt.is_ClassiForShape == 1:
				self.loss_names = self.loss_filter_multi_expan('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'Loc_D_fa', 'Loc_D_re', 'Loc_G_gan', 'Loc_G_Shp')

			# initialize optimizers
			# optimizer G
			if opt.niter_fix_global > 0:
				import sys
				if sys.version_info >= (3, 0):
					finetune_list = set()
				else:
					from sets import Set
					finetune_list = Set()

				params_dict = dict(self.netG.named_parameters())
				params = []
				for key, value in params_dict.items():
					if key.startswith('model' + str(opt.n_local_enhancers)):
						params += [value]
						finetune_list.add(key.split('.')[0])
				print(
					'------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
				print('The layers that are finetuned are ', sorted(finetune_list))
			else:
				params = list(self.netG.parameters())
				# add for SPADE
				for ele in params:
					ele.requires_grad = True
			if self.gen_features:
				params += list(self.netE.parameters())
			if self.is_scGraph == 2:
				params += list(self.scEGCN.parameters())



			self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

			# optimizer D
			params = list(self.netD.parameters())
			if self.is_scGraph == 3 and self.opt.MultiExpanTimes > 0:
				params += list(self.MultiExpan_netD.parameters())
			self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


		#----------------------------------------------------------
		#My own apart
	def mask_variable(self, input, mask):
		if not (input.dim() == mask.dim()):
			mask = mask.unsqueeze(1)   #mask要去看看是否 0，1
		output = input * mask.repeat(1, input.size(1), 1, 1)
		return output

	def find_local_object(self, masked_res, SelectC):
		num_channel = masked_res.size()[1]
		if num_channel > 1:
			res = masked_res[:, SelectC, :, :]
		elif num_channel == 1:
			# masked_res = masked_res.int()
			# res = torch.eq(torch.clamp(masked_res, min=0, max=self.input_nc-1), SelectC)
			res = None
		return res


		#----------------------------------------------------------

	def encode_input(self, label_map, inst_map=None, real_image1=None, real_image2=None, feat_map=None, label_map2 = None, infer=False):
		if self.opt.label_nc == 0:
			input_label = label_map.data.cuda()
		else:
			# create one-hot vector for label map
			size = label_map.size()
			oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
			input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
			# label_map_RemoveNegOne = copy.deepcopy(label_map)
			# NegOne_Map = torch.eq(label_map_RemoveNegOne, -1)
			# label_map_RemoveNegOne[NegOne_Map] = self.input_nc
			# input_label = input_label.scatter_(1, label_map_RemoveNegOne.data.long().cuda(), 1.0)   #会报错 因为-1是没法索引的
			input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)  #如果要把形状先验信息放进来，很大可能要让labelmap与其合并
			if self.opt.data_type == 16:
				input_label = input_label.half()

			#考虑 one-hot vector for label map 2 (groundtruth label map)
			if label_map2 is not None:
				oneHot_size2 = (size[0], self.opt.output_nc, size[2], size[3])
				gt_label = torch.cuda.FloatTensor(torch.Size(oneHot_size2)).zero_()
				gt_label = gt_label.scatter_(1, label_map2.data.long().cuda(), 1.0)  # 如果要把形状先验信息放进来，很大可能要让labelmap与其合并
				if self.opt.data_type == 16:
					gt_label = gt_label.half()
				# gt_label = Variable(label_map2.data.cuda()) #从35通道变成1通道
			else:
				gt_label = None

		# get edges from instance map
		if not self.opt.no_instance:
			# inst_map_RemoveNegOne = copy.deepcopy(inst_map)
			# inst_map_RemoveNegOne[NegOne_Map] = self.input_nc
			# inst_map_RemoveNegOne = inst_map_RemoveNegOne.data.cuda()
			inst_map = inst_map.data.cuda()
			edge_map = self.get_edges(inst_map)                                         #可以考虑将mask掉的部分的边缘取消掉其的计算
			input_label = torch.cat((input_label, edge_map), dim=1)
		input_label = Variable(input_label, requires_grad=infer)

		# real images for training
		if real_image1 is not None:
			real_image1 = Variable(real_image1.data.cuda())
		if real_image2 is not None:
			real_image2 = Variable(real_image2.data.cuda())

		# instance map for feature encoding
		if self.use_features:
			# get precomputed feature maps
			if self.opt.load_features:
				feat_map = Variable(feat_map.data.cuda())
			if self.opt.label_feat:
				inst_map = label_map.cuda()

		return input_label, inst_map, real_image1, real_image2, feat_map, gt_label

	def discriminate(self, input_label, test_image, use_pool=False):
		input_concat = torch.cat((input_label, test_image.detach()), dim=1)
		if use_pool:
			fake_query = self.fake_pool.query(input_concat)
			return self.netD.forward(fake_query)
		else:
			return self.netD.forward(input_concat)


	def discriminate_multi_expan(self, input_label, test_image, use_pool=False):
		input_concat_set = []
		for i in range(len(input_label)):
			input_concat = torch.cat((input_label[i], test_image[i].detach()), dim=1)
			if use_pool:
				input_concat = self.fake_pool.query(input_concat)
			input_concat_set.append(input_concat)
		return self.MultiExpan_netD.forward(input_concat_set)

	def forward(self, InComLableMap, InComInstMap, SelectLocLayer, InComImg, ComImage, SelectedBox, pretrained_cls, SeleL2ProjL, SelectC, feat, ComLabelMap):
		# Encode Inputs
		input_label, inst_map, InComImg, ComImage, feat_map, com_label = self.encode_input(InComLableMap, InComInstMap, InComImg, ComImage, feat, ComLabelMap)
		#此时的input label和inst_map是不完整的



		# --------------------------------------------------------------
		# Below is added by Jianfeng He
		# if SelectLocLayer != None:
		# if self.opt.is_shapePrior:
		input_label = torch.cat((input_label, SelectLocLayer), dim=1)
		# Above is added by Jianfeng He
		# --------------------------------------------------------------

		# Fake Generation
		if self.use_features:
			if not self.opt.load_features:
				feat_map = self.netE.forward(ComImage, inst_map)
			input_concat = torch.cat((input_label, feat_map), dim=1)
		elif self.is_scGraph == 2:
			input_concat = input_label
		else:
			input_concat = input_label      #要将input_lable和sceneGraph的Embedding连接起来
		# fake_image = self.netG.forward(input_concat)  #此时此刻是fake的label map
		fake_image1 = self.netG.forward(input_concat)  # 此时此刻是fake的label map
		# fake_image = torch.nn.LogSoftmax(dim=1)(fake_image1)
		fake_image = torch.nn.Softmax(dim=1)(fake_image1)



		# fake_image_show = torch.clamp(fake_image.detach().int(), 0, self.input_nc-1)
		_, fake_image_show = torch.max(fake_image, dim=1, keepdim=True)


		#build multi Expansion part

		# fake_masked_syn_image = com_label #InComImg #和eccv不同，此处换成了label map
		# fake_masked_syn_image[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1] \
		# 	= fake_image[:, :,SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1]

		SelectLocLayer_oppo_m_cha = (torch.ones_like(SelectLocLayer) - SelectLocLayer).repeat(1, com_label.size(1), 1, 1)
		SelectLocLayer_ori_m_cha = SelectLocLayer.repeat(1, com_label.size(1), 1, 1)
		fake_masked_syn_image = com_label * SelectLocLayer_oppo_m_cha + fake_image * SelectLocLayer_ori_m_cha
		# fake_masked_syn_image_show = torch.clamp(fake_masked_syn_image.detach().int(), 0, self.input_nc-1)
		_, fake_masked_syn_image_show = torch.max(fake_masked_syn_image, dim=1, keepdim=True)
		# _, fake_whole_sym_label = torch.max(fake_image, dim=1, keepdim=True)  #全部都是生成的label图
		# _, fake_masked_sym_label = torch.max(fake_masked_syn_image, dim=1, keepdim=True)  #只有select区域是生成的label图
		# 计算两个重建loss，global 和 local
		# recon_gloabl_loss = self.L1Loss_Global(fake_whole_sym_label.float(), ComLabelMap)
		# recon_local_loss = self.L1Loss_Lobal(fake_masked_sym_label[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1].float()
		# 									 , ComLabelMap[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1])


		if self.opt.MultiExpanTimes > 0 or self.opt.is_ClassiForShape == 1:
			TemporaryFlag = False
			if self.opt.is_ClassiForShape == 1 and self.opt.MultiExpanTimes == 0:
				TemporaryFlag = True
				self.opt.MultiExpanTimes = 1
			fake_local_image_set = []
			origin_local_image_set = []
			input_label_local_set = []
			ComImage_local_set = []
			size_map = input_label.shape
			for i in range(self.opt.MultiExpanTimes):
				x_min_chosen = SelectedBox[0] - i * self.opt.MultiExpanRadius
				if x_min_chosen < 0:
					x_min_chosen = 0
				x_max_chosen = SelectedBox[1] + i * self.opt.MultiExpanRadius
				if x_max_chosen >= size_map[2]:
					x_max_chosen = size_map[2]
				y_min_chosen = SelectedBox[2] - i * self.opt.MultiExpanRadius
				if y_min_chosen < 0:
					y_min_chosen = 0
				y_max_chosen = SelectedBox[3] + i * self.opt.MultiExpanRadius
				if y_max_chosen >= size_map[3]:
					y_max_chosen = size_map[3]
				fake_mid = fake_masked_syn_image[:, :, x_min_chosen:x_max_chosen+1, y_min_chosen:y_max_chosen+1]
				# origin_mid = com_label[:, :, x_min_chosen:x_max_chosen, y_min_chosen:y_max_chosen]
				input_label_local_mid = input_label[:, :, x_min_chosen:x_max_chosen+1, y_min_chosen:y_max_chosen+1]
				ComImage_local_mid = com_label[:, :, x_min_chosen:x_max_chosen+1, y_min_chosen:y_max_chosen+1]
				fake_local_image_set.append(fake_mid)   #第0个是完全卡死时候的情形
				# origin_local_image_set.append(origin_mid)
				input_label_local_set.append(input_label_local_mid)
				ComImage_local_set.append(ComImage_local_mid)
			if TemporaryFlag == True:
				self.opt.MultiExpanTimes = 0
				TemporaryFlag = False

		#ECCV Multi-Expan Loss
		#------------------------------
		loss_D_fa_local = torch.tensor(0)
		loss_D_re_local = torch.tensor(0)
		loss_G_loc_GAN = torch.tensor(0)
		if self.opt.MultiExpanTimes > 0:
			# Fa_Local Detection and Loss
			pred_fa_local_pool = self.discriminate_multi_expan(input_label_local_set, fake_local_image_set, use_pool=True)
			loss_D_fa_local = self.criterionGAN(pred_fa_local_pool, False)

			#Re_Local Detection and Loss
			pred_re_local_pool = self.discriminate_multi_expan(input_label_local_set, ComImage_local_set, use_pool=True)  #得设置成完整的对应完整的 但是 但是 但是 我自己设置的是不全的图片
			loss_D_re_local = self.criterionGAN(pred_re_local_pool, True)   # 从D角度来看，计算真实图片能够 real的损失

			# Local_Gan loss

			input_concat_set = []
			for i in range(len(input_label_local_set)):
				input_concat = torch.cat((input_label_local_set[i], fake_local_image_set[i]), dim=1)
				# if use_pool:
				# 	input_concat = self.fake_pool.query(input_concat)
				input_concat_set.append(input_concat)
			pred_fa_local = self.MultiExpan_netD.forward(input_concat_set)
			loss_G_loc_GAN = self.criterionGAN(pred_fa_local, True)  # 从G角度来看，计算生成图片能够 real的损失
		#------------------------------
		# ECCV Multi-Expan Loss

		#ECCV Shape Classification Loss
		#-----------------------------------------
		c_errG = torch.tensor(0)
		if self.opt.is_ClassiForShape == 1:
			ProjL = torch.tensor([SeleL2ProjL[str(SelectC)]]).cuda().long()
			# fake_l_shape = self.get_edges(fake_local_image_set[0])
			fake_l_shape = self.pt_get_edges(fake_local_image_set[0])    #当前是laplacian算子
			c_errG = self.cls_criterion(pretrained_cls.model(fake_l_shape), Variable(ProjL))
			# print('choose Lable is:',SeleL2ProjL[str(SelectC)], 'c_errG is:', c_errG)

		#-----------------------------------------
		#ECCV Shape Classification Loss

		# Fake Detection and Loss
		pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
		loss_D_fake = self.criterionGAN(pred_fake_pool, False)

		# Real Detection and Loss
		pred_real = self.discriminate(input_label, com_label)    #得设置成完整的对应完整的 但是 但是 但是 我自己设置的是不全的图片
		loss_D_real = self.criterionGAN(pred_real, True)  # 从D角度来看，计算真实图片能够 real的损失

		# GAN loss (Fake Passability Loss)
		pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
		loss_G_GAN = self.criterionGAN(pred_fake, True)  # 从G角度来看，计算生成图片能够 real的损失

		# GAN feature matching loss
		loss_G_GAN_Feat = 0
		if not self.opt.no_ganFeat_loss:
			feat_weights = 4.0 / (self.opt.n_layers_D + 1)  # self.opt.n_layers_D=3，但是层数却感觉是5 ？？？？
			D_weights = 1.0 / self.opt.num_D
			for i in range(self.opt.num_D):
				for j in range(len(pred_fake[i]) - 1):
					loss_G_GAN_Feat += D_weights * feat_weights * \
					                   self.criterionFeat(pred_fake[i][j], pred_real[i][
						                   j].detach()) * self.opt.lambda_feat  # 从二者一起的角度来看，在做feature matching

		# VGG feature matching loss
		loss_G_VGG = torch.tensor(0).float().cuda()  # 计算loss:perception loss，在这个版本中用于计算重建损失
		# if not self.opt.no_vgg_loss:
		# 	loss_G_VGG = self.criterionVGG(fake_image, com_label) * self.opt.lambda_feat
		if not self.opt.no_vgg_loss:
			if self.opt.global_mask == 1:
				comb_gt_mask = torch.ones_like(SelectLocLayer).detach()
			else:
				comb_gt_mask = SelectLocLayer.detach()

			# gl/l/ocal 全局中的只有待修改的 上下文ctx
			# loss_recon_comb = self.criterionCtxRecon(fake_image, ComLabelMap.long().squeeze(1), comb_gt_mask)
			loss_recon_comb = self.criterionCtxRecon(fake_image, ComLabelMap.long().squeeze(1), SelectLocLayer)
			#改成单通道后，这个Ctx的计算就有问题；而且显示的结果为全黑，这个问题也要解决


			#现在只做gl/l/ocal 全局中的只有待修改的obj loss
			# obj_generate_part1 = self.find_local_object(fake_image, SelectC) #此时的fake_image是标准化的结果
			# obj_generate_part2 = self.mask_variable(obj_generate_part1.unsqueeze(0), SelectLocLayer)
			#
			# obj_gt_part1 = self.find_local_object(com_label, SelectC)
			# obj_gt_part2 = self.mask_variable(obj_gt_part1.unsqueeze(0), SelectLocLayer)
			#
			# loss_recon_obj = self.criterionObjRecon(obj_generate_part2, obj_gt_part2)
			obj_generate_part1 = fake_image[:, SelectC, :, :]
			obj_generate_part2 = obj_generate_part1.unsqueeze(0).mul(SelectLocLayer)

			obj_gt_part1 = com_label[:, SelectC, :, :]
			obj_gt_part2 = obj_gt_part1.unsqueeze(0).mul(SelectLocLayer)

			loss_recon_obj = self.criterionObjRecon(obj_generate_part2, obj_gt_part2.detach())






			loss_G_VGG = loss_recon_comb + \
						 loss_recon_obj \
						 # + recon_gloabl_loss + recon_local_loss
			# loss_G_VGG = recon_gloabl_loss + recon_local_loss



		output_image = [fake_image_show, fake_masked_syn_image_show]
		# output_image = [fake_image, fake_masked_syn_image]

		# Only return the fake_B image if necessary to save BW
		# return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_rexal, loss_D_fake), fake_image]
		if self.opt.MultiExpanTimes <= 0 and self.opt.is_ClassiForShape==0:
			res = [self.loss_filter(loss_G_GAN.unsqueeze(0), loss_G_GAN_Feat.unsqueeze(0), loss_G_VGG.unsqueeze(0),
		                  loss_D_real.unsqueeze(0), loss_D_fake.unsqueeze(0)), output_image]
		elif self.opt.MultiExpanTimes > 0 or self.opt.is_ClassiForShape==1:
			res = [self.loss_filter_multi_expan(loss_G_GAN.unsqueeze(0), loss_G_GAN_Feat.unsqueeze(0), loss_G_VGG.unsqueeze(0),
		                  loss_D_real.unsqueeze(0), loss_D_fake.unsqueeze(0), loss_D_fa_local.unsqueeze(0), loss_D_re_local.unsqueeze(0), loss_G_loc_GAN.unsqueeze(0), c_errG.unsqueeze(0)), output_image]

		return res

	def forward_backup_RemovalSeg2RemovalImg(self, InComLableMap, InComInstMap, SelectLocLayer, InComImg, ComImage, feat):
		# Encode Inputs
		input_label, inst_map, InComImg, ComImage, feat_map = self.encode_input(InComLableMap, InComInstMap, InComImg, ComImage, feat)

		# --------------------------------------------------------------
		# Below is added by Jianfeng He

		input_label = torch.cat((input_label, SelectLocLayer), dim=1)
		# Above is added by Jianfeng He
		# --------------------------------------------------------------

		# Fake Generation
		if self.use_features:
			if not self.opt.load_features:
				feat_map = self.netE.forward(ComImage, inst_map)
			input_concat = torch.cat((input_label, feat_map), dim=1)
		elif self.is_scGraph == 2:
			input_concat = input_label
		else:
			input_concat = input_label      #要将input_lable和sceneGraph的Embedding连接起来
		fake_image = self.netG.forward(input_concat)



		# Fake Detection and Loss
		pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
		loss_D_fake = self.criterionGAN(pred_fake_pool, False)

		# Real Detection and Loss
		pred_real = self.discriminate(input_label, InComImg)    #得设置成完整的对应完整的 但是 但是 但是 我自己设置的是不全的图片
		loss_D_real = self.criterionGAN(pred_real, True)  # 从D角度来看，计算真实图片能够 real的损失

		# GAN loss (Fake Passability Loss)
		pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
		loss_G_GAN = self.criterionGAN(pred_fake, True)  # 从G角度来看，计算生成图片能够 real的损失

		# GAN feature matching loss
		loss_G_GAN_Feat = 0
		if not self.opt.no_ganFeat_loss:
			feat_weights = 4.0 / (self.opt.n_layers_D + 1)  # self.opt.n_layers_D=3，但是层数却感觉是5 ？？？？
			D_weights = 1.0 / self.opt.num_D
			for i in range(self.opt.num_D):
				for j in range(len(pred_fake[i]) - 1):
					loss_G_GAN_Feat += D_weights * feat_weights * \
					                   self.criterionFeat(pred_fake[i][j], pred_real[i][
						                   j].detach()) * self.opt.lambda_feat  # 从二者一起的角度来看，在做feature matching

		# VGG feature matching loss
		loss_G_VGG = 0  # 计算loss:perception loss
		if not self.opt.no_vgg_loss:
			loss_G_VGG = self.criterionVGG(fake_image, ComImage) * self.opt.lambda_feat

		# Only return the fake_B image if necessary to save BW
		return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake), fake_image]

	def inference(self, InComLableMap, InComInstMap, SelectLocLayer, InComImg, ComImage, SelectedBox, feat):
		ComLableMap = None
		input_label, inst_map, InComImg, ComImage, feat_map, _ = self.encode_input(InComLableMap, InComInstMap, InComImg,
		                                                                        ComImage,feat, ComLableMap)

		# --------------------------------------------------------------
		# Below is added by Jianfeng He
		# if SelectLocLayer != None:
		input_label = torch.cat((input_label, SelectLocLayer.cuda()), dim=1)
		# Above is added by Jianfeng He
		# --------------------------------------------------------------

		# Fake Generation
		if self.use_features:
			if not self.opt.load_features:
				feat_map = self.netE.forward(ComImage, inst_map)
			input_concat = torch.cat((input_label, feat_map), dim=1)
		elif self.is_scGraph == 2:
			input_concat = input_label
		else:
			input_concat = input_label  # 要将input_lable和sceneGraph的Embedding连接起来
		fake_image = self.netG.forward(input_concat)


		# # build one hot lable map in mult channel
		# if self.opt.label_nc == 0:
		# 	input_label_ini = InComLableMap.data.cuda()
		# else:
		# 	# create one-hot vector for label map
		# 	size = InComLableMap.size()
		# 	oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
		# 	input_label_ini = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
		# 	# label_map_RemoveNegOne = copy.deepcopy(label_map)
		# 	# NegOne_Map = torch.eq(label_map_RemoveNegOne, -1)
		# 	# label_map_RemoveNegOne[NegOne_Map] = self.input_nc
		# 	# input_label = input_label.scatter_(1, label_map_RemoveNegOne.data.long().cuda(), 1.0)   #会报错 因为-1是没法索引的
		# 	input_label_ini = input_label_ini.scatter_(1, InComLableMap.data.long().cuda(), 1.0)  #如果要把形状先验信息放进来，很大可能要让labelmap与其合并
		# 	if self.opt.data_type == 16:
		# 		input_label_ini = input_label_ini.half()
		# torch.max(fake_masked_syn_image, dim=1, keepdim=True)
		# build multi Expansion part
		_, fake_image_show = torch.max(fake_image, dim=1, keepdim=True)


		fake_masked_syn_image_show = InComLableMap
		fake_masked_syn_image_show[:, :, SelectedBox[0]:SelectedBox[1]+1, SelectedBox[2]:SelectedBox[3]+1] = fake_image_show[:, :,
		                                                                                            SelectedBox[0]:
		                                                                                            SelectedBox[1]+1,
		                                                                                            SelectedBox[2]:
		                                                                                            SelectedBox[3]+1]

		res = [fake_image_show, fake_masked_syn_image_show]
		return res

	def sample_features(self, inst):
		# read precomputed feature clusters
		cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
		features_clustered = np.load(cluster_path, encoding='latin1').item()

		# randomly sample from the feature clusters
		inst_np = inst.cpu().numpy().astype(int)
		feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
		for i in np.unique(inst_np):
			label = i if i < 1000 else i // 1000
			if label in features_clustered:
				feat = features_clustered[label]
				cluster_idx = np.random.randint(0, feat.shape[0])

				idx = (inst == int(i)).nonzero()
				for k in range(self.opt.feat_num):
					feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
		if self.opt.data_type == 16:
			feat_map = feat_map.half()
		return feat_map

	def encode_features(self, image, inst):
		image = Variable(image.cuda(), requires_grad=True)
		feat_num = self.opt.feat_num
		h, w = inst.size()[2], inst.size()[3]
		block_num = 32
		feat_map = self.netE.forward(image, inst.cuda())
		inst_np = inst.cpu().numpy().astype(int)
		feature = {}
		for i in range(self.opt.label_nc):
			feature[i] = np.zeros((0, feat_num + 1))
		for i in np.unique(inst_np):
			label = i if i < 1000 else i // 1000
			idx = (inst == int(i)).nonzero()
			num = idx.size()[0]
			idx = idx[num // 2, :]
			val = np.zeros((1, feat_num + 1))
			for k in range(feat_num):
				val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
			val[0, feat_num] = float(num) / (h * w // block_num)
			feature[label] = np.append(feature[label], val, axis=0)
		return feature

	def get_edges(self, t):
		edge = torch.cuda.ByteTensor(t.size()).zero_()
		# edge = torch.zeros(t.size())
		edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
		edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
		edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
		edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
		if self.opt.data_type == 16:
			return edge.half()
		else:
			return edge.float()

	def save(self, which_epoch):
		self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
		self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
		if self.gen_features:
			self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

	def update_fixed_params(self):
		# after fixing the global generator for a number of iterations, also start finetuning it
		params = list(self.netG.parameters())
		if self.gen_features:
			params += list(self.netE.parameters())
		self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
		if self.opt.verbose:
			print('------------ Now also finetuning global generator -----------')

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_D.param_groups:
			param_group['lr'] = lr
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		if self.opt.verbose:
			print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr

	def pt_get_edges(self, t):
		# define laplacian
		fil = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float().cuda()  # 别忘了，pytorch里的是cross-correlation 而不是卷积操作，得上下颠倒
		fil2 = fil.expand(3, 3, 3, 3)
		t = torch.nn.functional.conv2d(t, fil2, stride=1, padding=1)
		# res = torch.clamp(res, -1, 1)
		# res2 = (res + 1) / 2
		# show_from_tensor(res2)
		return t


class InferenceModel(MyBasePix2PixHDModel):
	def forward(self, inp):
		label, inst = inp
		return self.inference(label, inst)


