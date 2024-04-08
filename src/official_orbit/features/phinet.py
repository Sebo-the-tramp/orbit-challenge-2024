import torch
import torch.nn as nn

import micromind as mm
from micromind.networks import PhiNet, XiNet

from src.official_orbit.features.efficientnet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    Conv2dDynamicSamePadding,
    Conv2dStaticSamePadding,
    calculate_output_image_size
)

import sys

VALID_MODELS = (
    'phinet'
)

class PhinetAdapter(mm.MicroMind):
    """Implements an image classification class. Provides support
    for timm augmentation and loss functions."""

    def __init__(self, hparams, pretrained, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)
        
        self.modules['feature_extractor'] = PhiNet(
            input_shape=(3,224,224),
            alpha=2.3,
            num_layers=7,
            beta=0.75,
            t_zero=5,
            compatibility=False,
            divisor=8,
            downsampling_layers=[5,7],
            return_layers=None,
            # classification-specific
            include_top=False,
            num_classes=0,
        )     

        

        print("Number of parameters for each module:")
        print(self.compute_params())

        print("Number of MAC for each module:")
        print(self.compute_macs((3,224,224)))

        if pretrained:        
        
            # Taking away the classifier from pretrained model
            pretrained_dict = torch.load('../../../pretrained/phinet/state_dict.pth_v2.tar', map_location=torch.device('cuda:0'))
            model_dict = {}
            for k, v in pretrained_dict.items():
                if "classifier" not in k:
                    model_dict[k] = v
                    
            self.modules['feature_extractor'].load_state_dict(model_dict)
            # backbone unfrozen
            # for _, param in self.modules['feature_extractor'].named_parameters():
            #     param.requires_grad = False

            self.modules['flattener'] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )  

        self.modules.to(device='cuda:0')


    def forward(self, batch, drop_connect_rate=None):
        """Computes forward step for image classifier.

        Arguments
        ---------
        batch : List[torch.Tensor, torch.Tensor]
            Batch containing the images and labels.

        Returns
        -------
        Predicted class and augmented class. : Tuple[torch.Tensor, torch.Tensor]
        """
        img = self._flatten(batch)

        x = self.modules["feature_extractor"](img)
        x = self.modules["flattener"](x)
        return x

    def setup_criterion(self):
        """Setup of the loss function based on augmentation strategy."""
        # setup loss function

        return None

    def compute_loss(self, pred, batch):
        """Sets up the loss function and computes the criterion.

        Arguments
        ---------
        pred : Tuple[torch.Tensor, torch.Tensor]
            Predicted class and augmented class.
        batch : List[torch.Tensor, torch.Tensor]
            Same batch as input to the forward step.

        Returns
        -------
        Cost function. : torch.Tensor
        """
        self.criterion = self.setup_criterion()

        # taking it from pred because it might be augmented
        return self.criterion(pred[0], pred[1])

    def parameters(self):

        return self.modules.parameters()
    
    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x
        

    @property
    def output_size(self):
        return 440

def phinet(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **override_params): 
    """
        Constructs an Phinet model.
    """
    assert batch_norm == 'basic', 'TaskNorm not implemented for EfficientNets'

    model = PhinetAdapter(override_params, pretrained)

    return model

#  WORKING but the forward fucntion is missing

# def phinet(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **override_params): 
#     """
#         Constructs an EfficientNet-b0 model.
#     """
#     assert batch_norm == 'basic', 'TaskNorm not implemented for EfficientNets'

#     # model_type = 'phinet'
#     # blocks_args, global_params = get_model_params(model_type, override_params)

#     modules = torch.nn.ModuleDict({})  # init empty modules dict

#     modules['feature_extractor'] = PhiNet(
#         input_shape=(3,224,224),
#         alpha=2.3,
#         num_layers=7,
#         beta=0.75,
#         t_zero=5,
#         compatibility=False,
#         downsampling_layers=[5,7],
#         return_layers=None,
#         divisor=8,
#         # classification-specific
#         include_top=False,
#         num_classes=0,
#     )     

#     if pretrained:        
        
#         # Taking away the classifier from pretrained model
#         pretrained_dict = torch.load('../../../pretrained/phinet/state_dict.pth_v2.tar')
#         model_dict = {}
#         for k, v in pretrained_dict.items():
#             if "classifier" not in k:
#                 model_dict[k] = v
                
#         modules['feature_extractor'].load_state_dict(model_dict)
#         # backbone unfrozen
#         # for _, param in self.modules['feature_extractor'].named_parameters():
#         #     param.requires_grad = False

#         modules['flattener'] = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten()
#         )                    
        
#     return modules