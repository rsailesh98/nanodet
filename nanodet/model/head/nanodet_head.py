import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from .gfl_head import GFLHead
from .anchor.anchor_target import multi_apply


class NanoDetHead(GFLHead):
    """
    Modified from GFL, use same loss functions but much lightweight convolution heads
    """

    def __init__(self,
                 num_classes,
                 loss,
                 input_channel,
                 stacked_convs=2,
                 octave_base_scale=5,
                 scales_per_octave=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 reg_max=16,
                 share_cls_reg=False,
                 activation='LeakyReLU',
                 **kwargs):
        self.share_cls_reg = share_cls_reg
        self.activation = activation
        super(NanoDetHead, self).__init__(num_classes,
                                          loss,
                                          input_channel,
                                          stacked_convs,
                                          octave_base_scale,
                                          scales_per_octave,
                                          conv_cfg,
                                          norm_cfg,
                                          reg_max,
                                          **kwargs)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.anchor_strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList([nn.Conv2d(self.feat_channels,
                                                self.cls_out_channels +
                                                4 * (self.reg_max + 1) if self.share_cls_reg else self.cls_out_channels,
                                                1,
                                                padding=0) for _ in self.anchor_strides])
        # TODO: if
        self.gfl_reg = nn.ModuleList([nn.Conv2d(self.feat_channels,
                                                4 * (self.reg_max + 1),
                                                1,
                                                padding=0) for _ in self.anchor_strides])
        self.linear = nn.Linear(8, 1)
        self.linear.weight.data = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape(1, -1)
        self.linear.bias.data.fill_(0)

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                DepthwiseConvModule(chn,
                                    self.feat_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    bias=self.norm_cfg is None,
                                    activation=self.activation))
            if not self.share_cls_reg:
                reg_convs.append(
                    DepthwiseConvModule(chn,
                                        self.feat_channels,
                                        3,
                                        stride=1,
                                        padding=1,
                                        norm_cfg=self.norm_cfg,
                                        bias=self.norm_cfg is None,
                                        activation=self.activation))

        return cls_convs, reg_convs

    def init_weights(self):
        for seq in self.cls_convs:
            for m in seq:
                normal_init(m.depthwise, std=0.01)
                normal_init(m.pointwise, std=0.01)
        for seq in self.reg_convs:
            for m in seq:
                normal_init(m.depthwise, std=0.01)
                normal_init(m.pointwise, std=0.01)
        bias_cls = -4.595  # 用0.01的置信度初始化
        for i in range(len(self.anchor_strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)
        print('Finish initialize Lite GFL Head.')

    def forward(self, feats):
        output = multi_apply(self.forward_single,
                             feats,
                             self.cls_convs,
                             self.reg_convs,
                             self.gfl_cls,
                             self.gfl_reg,
                             )
        if torch.onnx.is_in_onnx_export():
            assert len(output) == 2 and len(output[0]) == len(output[1])
            output_list = []
            for score_out, box_out in zip(output[0], output[1]):
                output_list.append(torch.cat((box_out, score_out), dim=2))
            output = torch.cat((output_list), dim=1)
        return output

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg):
        cls_feat = x
        reg_feat = x
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)
        if self.share_cls_reg:
            feat = gfl_cls(cls_feat)
            cls_score, bbox_pred = torch.split(feat, [self.cls_out_channels, 4 * (self.reg_max + 1)], dim=1)
        else:
            cls_score = gfl_cls(cls_feat)
            bbox_pred = gfl_reg(reg_feat)

        for shift in bbox_pred:
            # Calculate the center coordinates of the bounding box
            box_center_x = (shift['bbox'][0] + shift['bbox'][2]) / 2
            box_center_y = (shift['bbox'][1] + shift['bbox'][3]) / 2
    
            # Calculate the grid cell that the bounding box overlaps with
            grid_x = int(box_center_x / bbox_pred)
            grid_y = int(box_center_y / bbox_pred)
    
            # Calculate the center coordinates of the grid cell
            grid_center_x = (grid_x + 0.5) * bbox_pred
            grid_center_y = (grid_y + 0.5) * bbox_pred
    
            # Calculate the distance between the center of the bounding box and the center of the grid cell
            shift_x = grid_center_x - box_center_x
            shift_y = grid_center_y - box_center_y
    
            # Print the shift values
            print(f'Shift for detection {shift["class"]}: {shift_x}, {shift_y}')
        
        if torch.onnx.is_in_onnx_export():
            batchsize = x.data.size(0)
            cls_score = torch.sigmoid(cls_score).reshape(batchsize, self.num_classes, -1).permute(0, 2, 1)
            bbox_pred = bbox_pred.reshape(batchsize, (self.reg_max+1)*4, -1).permute(0, 2, 1)
            bbox_pred = F.argmax(bbox_pred.reshape(batchsize, -1, self.reg_max + 1), dim=2)
            bbox_pred = self.linear(bbox_pred).reshape(batchsize, -1, 4)
        return cls_score, bbox_pred

        
    def intersection_over_union(box_a, box_b):
  # Determine the coordinates of each of the two boxes
        xA = max(box_a['left'], box_b['left'])
        yA = max(box_a['top'], box_b['top'])
        xB = min(box_a['left']+box_a['width'], box_b['left']+box_b['width'])
        yB = min(box_a['top']+box_a['height'], box_b['top']+box_b['height'])
    
        # Calculate the area of the intersection area
        area_of_intersection = (xB - xA + 1) * (yB - yA + 1)
 
        # Calculate the area of both rectangles
        box_a_area = (box_a['width'] + 1) * (box_a['height'] + 1)
        box_b_area = (box_b['width'] + 1) * (box_b['height'] + 1)
        # Calculate the area of intersection divided by the area of union
        # Area of union = sum both areas less the area of intersection
        iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection)
 
        # Return the score
        return iou
