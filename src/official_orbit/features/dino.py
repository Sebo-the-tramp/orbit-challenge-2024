import torch
import torch.nn as nn

from torchsummary import summary


VALID_MODELS = (
    'vit_small'
)

class VitAdapter(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')


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

        x = self.model(img)
        return x

    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x
        

    @property
    def output_size(self):
        return 384


def dino_vit(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **override_params): 
    """
        Constructs an Phinet model.
    """
    assert batch_norm == 'basic', 'TaskNorm not implemented for EfficientNets'

    print("DAIII CHE CI SEIIII")

    model = VitAdapter()
    # should be just correct like this because the head is the identity so nothing changes -> dimension is 384
    print(model)

    print(next(model.parameters()).device)

    # summary(model, input_size=(3,224,224), device='cpu')

    return model
