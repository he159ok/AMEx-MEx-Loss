import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'MyScPix2PixHDModel':
        from .My_sc_pix2pixHD_model import MyScPix2PixHDModel, MyScInferenceModel
        if opt.isTrain:
            model = MyScPix2PixHDModel()
        else:
            model = MyScInferenceModel()
    elif opt.model == 'MyBasePix2PixHDModel':
        from .My_base_pix2pixHD_model import MyBasePix2PixHDModel, MyBasePix2PixHDModel
        if opt.isTrain:
            model = MyBasePix2PixHDModel()
        else:
            model = MyBasePix2PixHDModel()
    elif opt.model == 'My_approximate_base_pix2pixHD_model':
        from .My_approximate_base_pix2pixHD_model import MyBasePix2PixHDModel, MyBasePix2PixHDModel
        if opt.isTrain:
            model = MyBasePix2PixHDModel()
        else:
            model = MyBasePix2PixHDModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        model = model.cuda(opt.gpu_ids[0])

    return model
