import timm

import torch
from torch import nn

VALID_MODELS = (
    'mobilevitv2_075'
)

class MobileVitAdapter(nn.Module):
    """Implements an image classification class. Provides support
    for timm augmentation and loss functions."""

    def __init__(self, hparams, pretrained, *args, **kwargs):
        super().__init__()
        
        self.model = timm.create_model(
            'mobilevitv2_075.cvnets_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )

        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)       

        # not sure if it's needed
        # self.model.to(device='cuda:0')

    def extract_features(self, inputs):

        output = self.model.forward_features(inputs)
        # output is unpooled, a (1, 384, 8, 8) shaped tensor

        output = self.model.forward_head(output, pre_logits=True)

        return output


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

        output = self.extract_features(img)

        
        # output is a (1, num_features) shaped tensor
        return output
    
    # def _change_in_channels(self, in_channels):
    #     """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
    #     Args:
    #         in_channels (int): Input data's channel number.
    #     """
    #     if in_channels != 3:
    #         Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
    #         out_channels = round_filters(32, self._global_params)
    #         self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x
        
    def parameters(self):
        return self.model.parameters()

    @property
    def output_size(self):        
        return 384

def mobilevitv2_075(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **override_params): 
    """
        Constructs an Phinet model.
    """
    assert batch_norm == 'basic', 'TaskNorm not implemented for EfficientNets'

    model = MobileVitAdapter(override_params, pretrained)

    return model