python MyTrain.py \
--name lable2city_128p_Full_one_channel_no_novel_loss \
--model MyBasePix2PixHDModel \
--is_scGraph 3 \
--label_nc 34 \
--dataroot ./datasets2/cityscapes/  \
--is_shapePrior 0 \
--is_scGraph 3 \
--niter 100 \
--niter_decay 100 \
--ImageFileEnd _img2labelcolor \
--MultiExpanTimes 0 \
--MultiExpanRadius 0 \
--is_ClassiForShape 0 \
--gpu_ids 1 \
--output_nc 34 \
--nip18model 0