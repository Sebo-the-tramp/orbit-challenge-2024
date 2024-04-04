import torch
import torch.nn as nn

import micromind as mm
from micromind.networks import PhiNet, XiNet

import sys

VALID_MODELS = (
    'phinet'
)

# class PhinetAdapter(mm.MicroMind):
#     """Implements an image classification class. Provides support
#     for timm augmentation and loss functions."""

#     def __init__(self, hparams, *args, **kwargs):
#         super().__init__(hparams, *args, **kwargs)
        
#         self.modules['classifier'] = PhiNet(
#             input_shape=(3,224,224),
#             alpha=2.3,
#             num_layers=7,
#             beta=0.75,
#             t_zero=5,
#             compatibility=False,
#             divisor=8,
#             downsampling_layers=[5,7],
#             return_layers=None,
#             # classification-specific
#             include_top=False,
#             num_classes=0,
#         )     

#         print("Number of parameters for each module:")
#         print(self.compute_params())

#         print("Number of MAC for each module:")
#         print(self.compute_macs(hparams.input_shape))

#     def forward(self, batch):
#         """Computes forward step for image classifier.

#         Arguments
#         ---------
#         batch : List[torch.Tensor, torch.Tensor]
#             Batch containing the images and labels.

#         Returns
#         -------
#         Predicted class and augmented class. : Tuple[torch.Tensor, torch.Tensor]
#         """
#         img, target = batch

#         return (self.modules["classifier"](img), target)


    
# def get_model_params(model_name, override_params):
#     """Get the block args and global params for a given model name.
#     Args:
#         model_name (str): Model's name.
#         override_params (dict): A dict to modify global_params.
#     Returns:
#         blocks_args, global_params
#     """
#     blocks_args, global_params = 100000, 100000
#     return blocks_args, global_params
    
# def phinet(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **override_params): 
#     """
#         Constructs an EfficientNet-b0 model.
#     """
#     assert batch_norm == 'basic', 'TaskNorm not implemented for EfficientNets'

#     # model_type = 'phinet'
#     # blocks_args, global_params = get_model_params(model_type, override_params)

#     model = PhinetAdapter(override_params)

#     modules['classifier'] = PhiNet(
#         input_shape=(3,224,224),
#         alpha=2.3,
#         num_layers=7,
#         beta=0.75,
#         t_zero=5,
#         compatibility=False,
#         divisor=8,
#         downsampling_layers=[5,7],
#         return_layers=None,
#         # classification-specific
#         include_top=False,
#         num_classes=0,
#     )     

#     if pretrained:        
        
#         # Taking away the classifier from pretrained model
#         pretrained_dict = torch.load('../../../pretrained/phinet/state_dict.pth.tar', map_location='cuda:0')
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

# def forward(self, batch):
#         """Computes forward step for image classifier.

#         Arguments
#         ---------
#         batch : List[torch.Tensor, torch.Tensor]
#             Batch containing the images and labels.

#         Returns
#         -------
#         Predicted class and augmented class. : Tuple[torch.Tensor, torch.Tensor]
#         """
#         img, target = batch
#         if not self.hparams.prefetcher:
#             img, target = img.to(self.device), target.to(self.device)
#             if self.mixup_fn is not None:
#                 img, target = self.mixup_fn(img, target)

#         return (self.modules["classifier"](img), target)


def phinet(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **override_params): 
    """
        Constructs an EfficientNet-b0 model.
    """
    assert batch_norm == 'basic', 'TaskNorm not implemented for EfficientNets'

    # model_type = 'phinet'
    # blocks_args, global_params = get_model_params(model_type, override_params)

    modules = torch.nn.ModuleDict({})  # init empty modules dict

    modules['feature_extractor'] = PhiNet(
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

    if pretrained:        
        
        # Taking away the classifier from pretrained model
        pretrained_dict = torch.load('../../../pretrained/phinet/state_dict.pth.tar')
        model_dict = {}
        for k, v in pretrained_dict.items():
            if "classifier" not in k:
                model_dict[k] = v
                
        modules['feature_extractor'].load_state_dict(model_dict)
        # backbone unfrozen
        # for _, param in self.modules['feature_extractor'].named_parameters():
        #     param.requires_grad = False

        modules['flattener'] = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )                    
        
    return modules