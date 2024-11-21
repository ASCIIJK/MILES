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
        model = timm.create_model("hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == 'swin-b-p4-w7-in22k-ft-in1k':
        model = timm.create_model("hf_hub:timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == 'convnext-b-in22k-ft-in1k':
        model = timm.create_model("hf_hub:timm/convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True, num_classes=0)
        model.out_dim = 1024
    elif name == 'mambaout-base-in12k':
        model = timm.create_model("hf_hub:timm/mambaout_base_plus_rw.sw_e150_in12k", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == "caformer-b36-in22k":
        model = timm.create_model("hf_hub:timm/caformer_b36.sail_in22k", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == "convformer-b36-in22k":
        model = timm.create_model("hf_hub:timm/convformer_b36.sail_in22k", pretrained=True, num_classes=0)
        model.out_dim = 768
    elif name == "maxxvitv2-base-in12k":
        model = timm.create_model("hf_hub:timm/maxxvitv2_rmlp_base_rw_224.sw_in12k", pretrained=True, num_classes=0)
        model.out_dim = 1024
    else:
        raise 'No this type backbone!'

    return model.eval()
