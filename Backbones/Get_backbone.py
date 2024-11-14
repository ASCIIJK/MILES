from torch import nn
import torch
import timm


def getbackbone(name):
    if name == "pretrained_vit_b16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == 'vit-b-p16-in1k':
        model = timm.create_model("hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == 'vit-b-p16-in12k-in1k-clip':
        model = timm.create_model("vit_base_patch16_clip_224", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == 'swin-b-p4-w7-in22k-ft-in1k':
        model = timm.create_model("swinv2_base_window12to16_192to256", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == 'convnext-b-in22k-ft-in1k':
        model = timm.create_model("convnextv2_base", pretrained=True, num_classes=0)
        model.out_dim = 1024
    elif name == 'vit-l-p16-in21k':
        model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=0)
        model.out_dim = 1024
    elif name == 'vit-h-p16-in21k':
        model = timm.create_model("vit_huge_patch14_224", pretrained=True, num_classes=0)
        model.out_dim = 1024
    elif name == 'maxvit-base-in21k':
        model = timm.create_model("maxvit_base_tf_224", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == 'mambaout-base-in12k':
        model = timm.create_model("hf_hub:timm/mambaout_base_plus_rw.sw_e150_in12k", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == "caformer-s36-in22k":
        model = timm.create_model("caformer_s36", pretrained=True, num_classes=0)
        model.out_dim = 512
    elif name == "maxxvitv2-base-in12k":
        model = timm.create_model("maxxvitv2_rmlp_base_rw_224", pretrained=True, num_classes=0)
        model.out_dim = 1024
    else:
        raise 'No this type backbone!'

    return model.eval()
