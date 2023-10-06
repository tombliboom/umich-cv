"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        """
        
        For dummy input images with shape: (2, 3, 224, 224)
            Shape of c3 features: torch.Size([2, 64, 28, 28])
            Shape of c4 features: torch.Size([2, 160, 14, 14])
            Shape of c5 features: torch.Size([2, 400, 7, 7])
        
        """
        for level_name, feature_shape in dummy_out_shapes:
            _, in_channels, _, _ = feature_shape
            self.fpn_params[level_name] = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels,
                                                    kernel_size=1)
        out_layers = ['p5', 'p4', 'p3']
        for out_layer in out_layers:
            self.fpn_params[out_layer] = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                                                   kernel_size=3, padding=1, stride=1)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multiscale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        backbone_features = {}
        last_layer_info = None
        out_layers = ['p5', 'p4', 'p3']
        strides_dict = self.fpn_strides
        feature_layers = ['c5', 'c4', 'c3']

        for level_name in feature_layers:
            backbone_feature = backbone_feats[level_name]
            backbone_features[level_name] = self.fpn_params[level_name](backbone_feature)

        for feature_layer, out_layer in zip(feature_layers, out_layers):
            if last_layer_info is None:
                fpn_feats[out_layer] = backbone_features[feature_layer]
                last_layer_info = (out_layer, fpn_feats[out_layer])
            else:
                last_layer, last_layer_out = last_layer_info
                scale_factor = strides_dict[last_layer] // strides_dict[out_layer]
                last_layer_out = F.interpolate(last_layer_out, scale_factor=scale_factor)
                fpn_feats[out_layer] = backbone_features[feature_layer] + last_layer_out
                last_layer_info = (out_layer, fpn_feats[out_layer])

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
        shape_per_fpn_level: Dict[str, Tuple],
        strides_per_fpn_level: Dict[str, int],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        B, C, F_H, F_W = feat_shape
        coordinates = torch.zeros(size=(F_H, F_H, 2), dtype=dtype, device=device)
        for h in range(F_H):
            for w in range(F_W):
                cx, cy = level_stride * (h + 0.5), level_stride * (w + 0.5)
                coordinates[h, w, 0] = cx
                coordinates[h, w, 1] = cy
        location_coords[level_name] = coordinates.reshape(-1, 2)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shape (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the intering ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes inter, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    if boxes.numel() == 0:
        return keep

    keep = []  # a list, convert to python long at last
    x1, y1, x2, y2 = boxes[:, :4].unbind(dim=1)
    area = torch.mul(x2 - x1, y2 - y1)  # area of each boxes
    _, index = scores.sort(0)  # sort the score in ascending order

    count = 0
    while index.numel() > 0:
        # keep the highest-scoring box and remove that from the index list
        largest_idx = index[-1]
        keep.append(largest_idx)
        count += 1
        index = index[:-1]

        # if no more box remaining, break
        if index.size(0) == 0:
            break

        # get the x1,y1,x2,y2 of all the remaining boxes, and clamp them so that
        # we get the coord of intersection of boxes and highest-scoring box
        x1_inter = torch.index_select(x1, 0, index).clamp(min=x1[largest_idx])
        y1_inter = torch.index_select(y1, 0, index).clamp(min=y1[largest_idx])
        x2_inter = torch.index_select(x2, 0, index).clamp(max=x2[largest_idx])
        y2_inter = torch.index_select(y2, 0, index).clamp(max=y2[largest_idx])

        # clamp the width and height, get the intersect area
        W_inter = (x2_inter - x1_inter).clamp(min=0.0)
        H_inter = (y2_inter - y1_inter).clamp(min=0.0)
        inter_area = W_inter * H_inter

        # retrieve the areas of all the remaining boxes, and get the union area
        areas = torch.index_select(area, 0, index)
        union_area = (areas - inter_area) + area[largest_idx]

        # keep the boxes that have IoU <= iou_threshold
        IoU = inter_area / union_area
        index = index[IoU.le(iou_threshold)]

    # convert list to torch.long
    keep = torch.Tensor(keep).to(device=scores.device).long()
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep

def class_spec_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch. Long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
