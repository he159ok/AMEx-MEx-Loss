# Code Introduction
This is the code for *Neurocomputing 2022* paper [Semantic Inpainting on Segmentation Map via Multi-Expansion Loss](https://www.sciencedirect.com/science/article/abs/pii/S0925231222007433)

Folder "AMEx_Loss" includes key files for AMEx loss on natural image inpaitning.

Folder "semantic_editing" includes the implementation of 3-channel SISM task. It includes both MEx loss and AMEx loss on SISM.

## AMEx Loss on natural image processing
The AMEx Loss are implmented in `AMEx_Loss/net_gl_mex.py->_netlocalD.py`. 

The original Global and Local GAN loss is in `AMEx_Loss/net_gl.py->_netlocalD.py`.

The `AMEx_Loss/MyTrain` includes how the AMEx is used in the optimizer.

## AMEx Loss & MEx Loss on Semantic Inpainting on Segmentation Map (SISM)

The environment is in `semantic_editing/py35pt041/yaml`.

### 1. For cityscape

1.1 For Pipeline,

`python MyTrain.py --name lable2city_128p_Full_NonMultiExp --model MyBasePix2PixHDModel --is_scGraph 1 --label_nc 36 --dataroot ./datasets2/cityscapes/ --is_shapePrior 0 --is_scGraph 3 --niter 100 --niter_decay 100 --ImageFileEnd _img2labelcolor --MultiExpanTimes 0 --MultiExpanRadius 5 --gpu_ids 0 --loadSize 256 --fineSize 128 --labmdaShape 1 --labmdaMulExp 1`

1.2 For Pipeline + MultiExpansion,

`python MyTrain.py --name lable2city_128p_Full_OnlyMultiEx --model MyBasePix2PixHDModel --is_scGraph 1 --label_nc 36 --dataroot ./datasets2/cityscapes/ --is_shapePrior 0 --is_scGraph 3 --niter 100 --niter_decay 100 --ImageFileEnd _img2labelcolor --MultiExpanTimes 4 --MultiExpanRadius 5 --gpu_ids 0 --loadSize 256 --fineSize 128 --labmdaShape 1 --labmdaMulExp 1`

1.3 For evaluation (no matter use A-MEX/MEx loss or not):

`python MyTest_backup_0902.py --Te --name lable2city_128p_Full_OnlyMultiEx --model MyBasePix2PixHDModel --which_epoch 100 --netG generator_from_SPADE --is_scGraph 3 --label_nc 36 --dataroot ./datasets2/cityscapes/ --ImageFileEnd _img2labelcolor --how_many 501 --gpu_ids 0 --is_shapePrior 0 --is_ClassiForShape 0 --loadSize 256 --fineSize 128`

