# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .resnet import resnet18, resnet18_84
from .efficientnet import efficientnetb0
from .mobilevit import mobilevit_v2_14_224
from .mobilevitb import mobilevitv2_075

extractors = {
        'resnet18': resnet18,
        'resnet18_84': resnet18_84,
        'efficientnetb0' : efficientnetb0,
        'mobilevit': mobilevit_v2_14_224,
        'mobilevitb' : mobilevitv2_075
        }