import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)


def model_getter(device, image_size, model_type:str='sensor_3d', pretrained_model_dir=None, kernel_size:int=3):
    ## 3 modes [ sensor3D Unet, Attention c sensor3D Unet, Domain Adaptation c Attention c sensor3D Unet ]

    # Attention
    if model_type == 'attention':
        from model.Attention_Sensor3d import DeepSequentialNet
        print("###### Sensor3D + Attention model ######")
        sqNet = DeepSequentialNet(image_size, device).to(device)
        sqNet.apply(weights_init)
    # just sensor3D Unet
    elif model_type == 'sensor_3d':
        from model.Sensor3d import DeepSequentialNet
        print("###### Sensor3D model ######")
        sqNet = DeepSequentialNet(image_size, device).to(device)
        sqNet.apply(weights_init)

    # Domain Adaptation, Attention
    elif model_type == 'transfer_learning':
        from model.Attention_Sensor3d import DeepSequentialNet
        print("###### Sensor3D + Attention model + transfer learning ######")
        sqNet = DeepSequentialNet(image_size, device).to(device)
        sqNet.load_state_dict(torch.load(pretrained_model_dir, map_location=device))

    # SA_Unet
    elif model_type == 'sa_unet':
        from model.Torch_SA_Unet_OpthalCT import SA_Unet_Torch
        sqNet = SA_Unet_Torch(image_size, device,n_class=1,kernel_size=kernel_size).to(device)
        sqNet.apply(weights_init)
        print("###### Sensor3D + SA Unet Model ######")

    return sqNet


