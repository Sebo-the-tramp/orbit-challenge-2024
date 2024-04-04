# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .resnet import resnet18, resnet18_84
from .efficientnet import efficientnetb0
from .phinet import phinet

extractors = {
        'resnet18': resnet18,
        'resnet18_84': resnet18_84,
        'efficientnetb0' : efficientnetb0,
        'phinet': phinet,
        }
