# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .resnet import resnet18, resnet18_84
from .efficientnet import efficientnetb0
from .mobilevitb import mobilevitv2_075
from .phinet import phinet
from .dino import dino_vit

extractors = {
        'resnet18': resnet18,
        'resnet18_84': resnet18_84,
        'efficientnetb0' : efficientnetb0,
        'mobilevitb' : mobilevitv2_075,
        'phinet': phinet,
        'vit_small': dino_vit
        }
