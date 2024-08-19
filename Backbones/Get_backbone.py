from torch import nn
import timm


def getbackbone(name):
    if name == "pretrained_vit_b16_224_in21k":
        pretrained_cfg_overlay = dict(
            file="./pre_trained_backbone/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz")
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0,
                                  pretrained_cfg_overlay=pretrained_cfg_overlay)
        model.out_dim = 768
    else:
        raise 'No this type backbone!'

    return model
