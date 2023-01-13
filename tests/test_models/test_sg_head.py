from nanodet.model.seg_head import build_seg_head
from nanodet.model.backbone import build_backbone
import torch

def test_fastscnnneck():
    cfg=dict(
        name="FastSCNNNeck",
        in_channels=[24,320],
        feat_channels=[128,128],
        indexes=[-1,0],
        
    )
    neck=build_seg_head(cfg)
    print(neck)
    backbone_cfg=dict(
        name="EfficientNetLite",
        model_name="efficientnet_lite2",
        out_stages=(1, 2, 4, 6),
        activation="LeakyReLU"
    )
    #backbone=build_backbone(backbone_cfg)
    neck_cfg=dict(
        name="GhostPAN",
        in_channels=[48, 120, 352],
        out_channels=128,
        kernel_size=5,
    )
    dummy_1=torch.rand(3,24,64,64)
    dummy_2=torch.rand(3,320,16,16)
    dummy_input=[dummy_1,None,None,dummy_2]
    output=neck(dummy_input)
    print(output[-1].shape)