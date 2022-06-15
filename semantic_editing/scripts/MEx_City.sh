#conda actiate py35pt041
dname=$(dirname "$PWD")
cd $dname
python MyTrain.py --name cityscape_OnlyMultiExp_q4r5 --model MyBasePix2PixHDModel --label_nc 36 --dataroot ./datasets2/cityscapes/ --dataName cityscape --is_shapePrior 0 --is_scGraph 3 --niter 100 --niter_decay 100 --ImageFileEnd _img2labelcolor_MyOwn --MultiExpanTimes 4 --MultiExpanRadius 5 --is_ClassiForShape 0 --gpu_ids 1 --labmdaShape 1 --loadSize 256 --fineSize 128


/home/jfhe/anaconda3/envs/py35pt041/bin/python -u /home/jfhe/Documents/MountHe/jfhe/Project/MyTrain.py --name cityscape_OnlyMultiExp_q4r5 --model MyBasePix2PixHDModel --label_nc 36 --dataroot ./datasets2/cityscapes/ --dataName cityscape --is_shapePrior 0 --is_scGraph 3 --niter 100 --niter_decay 100 --ImageFileEnd _img2labelcolor_MyOwn --MultiExpanTimes 4 --MultiExpanRadius 5 --is_ClassiForShape 0 --gpu_ids 1 --labmdaShape 1 --loadSize 256 --fineSize 128
